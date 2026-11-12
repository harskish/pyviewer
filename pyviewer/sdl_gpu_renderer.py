"""imgui_bundle draw-data renderer using PySDL3's SDL_GPU API.

Vulkan/SPIR-V on Linux and Windows, Metal/MSL on macOS. Composites in linear
Rec.709 RGBA16F, then encodes SDR sRGB, extended-linear scRGB, or BT.2020/PQ.
Standard rendering uses stock SDL3/PySDL3; Torch interop uses an opt-in SDL bridge.
"""

import ctypes
import time
import warnings
from pathlib import Path

import numpy as np
from .sdl3_backend import SDL3Platform
import sdl3 as sdl
from . import sdl_torch_interop
from imgui_bundle import imgui


def checked(value):
    if not value:
        raise RuntimeError(sdl.SDL_GetError().decode())
    return value


def srgb_to_linear(value):
    value = np.asarray(value)
    return np.where(value <= 0.04045, value / 12.92, ((value + 0.055) / 1.055) ** 2.4)


class GPUTexture:
    def __init__(self, renderer, width, height, *, float_texture=True, srgb=False, pq=False, target=False):
        self.renderer = renderer
        self.width, self.height = width, height
        self.srgb = srgb
        self.pq = pq
        self.nearest = False
        self.dtype = np.float16 if float_texture else np.uint8
        self.pending = None
        self.pending_tensor = None
        self.pending_ready = None
        self.native_metal_texture = None
        self.cuda_transfer = None
        self.native_pending_copy = False
        self.uploaded = False
        self.format = (sdl.SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT if float_texture
                       else sdl.SDL_GPU_TEXTUREFORMAT_R8G8B8A8_UNORM)
        info = sdl.SDL_GPUTextureCreateInfo(
            type=sdl.SDL_GPU_TEXTURETYPE_2D, format=self.format,
            usage=sdl.SDL_GPU_TEXTUREUSAGE_SAMPLER | (sdl.SDL_GPU_TEXTUREUSAGE_COLOR_TARGET if target else 0),
            width=width, height=height, layer_count_or_depth=1, num_levels=1,
            sample_count=sdl.SDL_GPU_SAMPLECOUNT_1,
        )
        props = sdl.SDL_CreateProperties()
        try:
            if renderer.native_interop and float_texture and not target:
                checked(sdl.SDL_SetBooleanProperty(props, b'pyviewer.interop', True))
            info.props = props
            self.texture = checked(sdl.SDL_CreateGPUTexture(renderer.device, ctypes.byref(info)))
            self.native_metal_texture = sdl.SDL_GetPointerProperty(props, b'pyviewer.metal.texture', None)
        finally:
            sdl.SDL_DestroyProperties(props)
        self.id = renderer._next_texture_id
        renderer._next_texture_id += 1
        renderer.textures[self.id] = self
        self.ref = imgui.ImTextureRef(self.id)

    def upload(self, rgba):
        assert rgba.shape == (self.height, self.width, 4)
        self.pending_tensor = None
        self.pending_ready = None
        self.pending = np.ascontiguousarray(rgba, dtype=self.dtype)
        self.uploaded = True

    def upload_torch(self, rgba, ready=None):
        """Queue a GPU tensor without reading its pixels on the CPU."""
        if self.renderer.interop_policy == 'off':
            return False
        if not sdl_torch_interop.supports_tensor(self.renderer, rgba):
            if rgba.device.type != 'cpu' and self.renderer.interop_policy == 'required':
                raise RuntimeError(f'No native Torch interop for {rgba.device} and this SDL driver')
            return False
        if not self.renderer.native_interop:
            self.renderer.interop_failure(RuntimeError('SDL native bridge is unavailable; use the interop-enabled SDL build'))
            return False
        assert rgba.shape == (self.height, self.width, 4)
        self.pending_tensor, self.pending_ready = rgba, ready
        self.pending = None
        self.uploaded = True
        return True

    def release(self):
        if self.cuda_transfer is not None or self.native_metal_texture:
            self.renderer.wait_native()
        if self.cuda_transfer is not None:
            self.cuda_transfer.release()
            self.cuda_transfer = None
        self.pending_tensor = self.pending_ready = None
        if self.texture:
            sdl.SDL_ReleaseGPUTexture(self.renderer.device, self.texture)
            self.renderer.textures.pop(self.id)
            self.texture = None
            self.pending = None


class SDLGPUImguiRenderer(SDL3Platform):
    def __init__(self, window, device, *, mode, sdr_white_nits=203.0, gpu_interop='auto', **kwargs):
        self.device = device
        self.interop_policy = gpu_interop
        self.native_interop = gpu_interop != 'off' and sdl_torch_interop.bridge_version(device) == 1
        self.last_upload_backend = 'cpu'
        self.last_native_upload_ms = 0.0
        self._interop_warning = False
        self._last_fence = None
        self.mode = mode
        self.sdr_white_nits = sdr_white_nits
        self.sc_rgb_white = 1.0
        self.textures = {}
        self._next_texture_id = 1
        self._font_textures = {}
        self._buffers = {}
        self.scene = None
        self.last_presented = False
        super().__init__(window, **kwargs)
        self.io.backend_flags |= (imgui.BackendFlags_.renderer_has_textures
                                  | imgui.BackendFlags_.renderer_has_vtx_offset)
        self._shaders = []
        self._ui_pipeline = self._pipeline('imgui', sdl.SDL_GPU_TEXTUREFORMAT_R16G16B16A16_FLOAT, True)
        self._present_pipeline = self._pipeline('present', sdl.SDL_GetGPUSwapchainTextureFormat(device, window), False)
        sampler_info = sdl.SDL_GPUSamplerCreateInfo(
            min_filter=sdl.SDL_GPU_FILTER_LINEAR, mag_filter=sdl.SDL_GPU_FILTER_LINEAR,
            mipmap_mode=sdl.SDL_GPU_SAMPLERMIPMAPMODE_NEAREST,
            address_mode_u=sdl.SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            address_mode_v=sdl.SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
            address_mode_w=sdl.SDL_GPU_SAMPLERADDRESSMODE_CLAMP_TO_EDGE,
        )
        self.sampler = checked(sdl.SDL_CreateGPUSampler(device, ctypes.byref(sampler_info)))
        sampler_info.min_filter = sampler_info.mag_filter = sdl.SDL_GPU_FILTER_NEAREST
        self.nearest_sampler = checked(sdl.SDL_CreateGPUSampler(device, ctypes.byref(sampler_info)))

    def _shader(self, name, vertex):
        root = Path(__file__).with_name('sdl_gpu_shaders')
        metal = sdl.SDL_GetGPUDeviceDriver(self.device) == b'metal'
        code = (root / (name + ('.msl' if metal else '.spv'))).read_bytes()
        memory = (ctypes.c_uint8 * len(code)).from_buffer_copy(code)
        info = sdl.SDL_GPUShaderCreateInfo(
            code_size=len(code), code=memory, entrypoint=b'main0' if metal else b'main',
            format=sdl.SDL_GPU_SHADERFORMAT_MSL if metal else sdl.SDL_GPU_SHADERFORMAT_SPIRV,
            stage=sdl.SDL_GPU_SHADERSTAGE_VERTEX if vertex else sdl.SDL_GPU_SHADERSTAGE_FRAGMENT,
            num_samplers=0 if vertex else 1,
            num_uniform_buffers=0 if name == 'present.vert' else 1,
        )
        shader = checked(sdl.SDL_CreateGPUShader(self.device, ctypes.byref(info)))
        self._shaders.append(shader)
        return shader

    def _pipeline(self, name, texture_format, ui):
        target = sdl.SDL_GPUColorTargetDescription(format=texture_format)
        if ui:
            target.blend_state = sdl.SDL_GPUColorTargetBlendState(
                src_color_blendfactor=sdl.SDL_GPU_BLENDFACTOR_SRC_ALPHA,
                dst_color_blendfactor=sdl.SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                color_blend_op=sdl.SDL_GPU_BLENDOP_ADD,
                src_alpha_blendfactor=sdl.SDL_GPU_BLENDFACTOR_ONE,
                dst_alpha_blendfactor=sdl.SDL_GPU_BLENDFACTOR_ONE_MINUS_SRC_ALPHA,
                alpha_blend_op=sdl.SDL_GPU_BLENDOP_ADD, enable_blend=True,
            )
        info = sdl.SDL_GPUGraphicsPipelineCreateInfo(
            vertex_shader=self._shader(name + '.vert', True),
            fragment_shader=self._shader(name + '.frag', False),
            primitive_type=sdl.SDL_GPU_PRIMITIVETYPE_TRIANGLELIST,
        )
        info.multisample_state.sample_count = sdl.SDL_GPU_SAMPLECOUNT_1
        info.rasterizer_state.fill_mode = sdl.SDL_GPU_FILLMODE_FILL
        info.rasterizer_state.cull_mode = sdl.SDL_GPU_CULLMODE_NONE
        info.target_info.color_target_descriptions = ctypes.pointer(target)
        info.target_info.num_color_targets = 1
        if ui:
            buffer = sdl.SDL_GPUVertexBufferDescription(slot=0, pitch=imgui.VERTEX_SIZE,
                                                       input_rate=sdl.SDL_GPU_VERTEXINPUTRATE_VERTEX)
            attributes = (sdl.SDL_GPUVertexAttribute * 3)(
                sdl.SDL_GPUVertexAttribute(location=0, buffer_slot=0,
                    format=sdl.SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, offset=imgui.VERTEX_BUFFER_POS_OFFSET),
                sdl.SDL_GPUVertexAttribute(location=1, buffer_slot=0,
                    format=sdl.SDL_GPU_VERTEXELEMENTFORMAT_FLOAT2, offset=imgui.VERTEX_BUFFER_UV_OFFSET),
                sdl.SDL_GPUVertexAttribute(location=2, buffer_slot=0,
                    format=sdl.SDL_GPU_VERTEXELEMENTFORMAT_UBYTE4_NORM, offset=imgui.VERTEX_BUFFER_COL_OFFSET),
            )
            info.vertex_input_state = sdl.SDL_GPUVertexInputState(
                vertex_buffer_descriptions=ctypes.pointer(buffer), num_vertex_buffers=1,
                vertex_attributes=attributes, num_vertex_attributes=3)
        return checked(sdl.SDL_CreateGPUGraphicsPipeline(self.device, ctypes.byref(info)))

    def _transfer(self, data, usage=sdl.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD):
        info = sdl.SDL_GPUTransferBufferCreateInfo(usage=usage, size=data.nbytes)
        transfer = checked(sdl.SDL_CreateGPUTransferBuffer(self.device, ctypes.byref(info)))
        if usage == sdl.SDL_GPU_TRANSFERBUFFERUSAGE_UPLOAD:
            ptr = checked(sdl.SDL_MapGPUTransferBuffer(self.device, transfer, False))
            ctypes.memmove(ptr, data.ctypes.data, data.nbytes)
            sdl.SDL_UnmapGPUTransferBuffer(self.device, transfer)
        return transfer

    def _update_imgui_textures(self):
        for tex in imgui.get_platform_io().textures:
            if tex.status == imgui.ImTextureStatus.want_create:
                assert tex.format == imgui.ImTextureFormat.rgba32
                texture = GPUTexture(self, tex.width, tex.height, float_texture=False, srgb=tex.use_colors)
                self._font_textures[texture.id] = texture
                tex.set_tex_id(texture.id)
                texture.upload(tex.get_pixels_array().reshape(tex.height, tex.width, 4))
                tex.set_status(imgui.ImTextureStatus.ok)
            elif tex.status == imgui.ImTextureStatus.want_updates:
                self.textures[tex.tex_id].upload(tex.get_pixels_array().reshape(tex.height, tex.width, 4))
                tex.set_status(imgui.ImTextureStatus.ok)
            elif tex.status == imgui.ImTextureStatus.want_destroy:
                self._font_textures.pop(tex.tex_id).release()
                tex.set_tex_id(0)
                tex.set_status(imgui.ImTextureStatus.destroyed)

    def interop_failure(self, error):
        if self.interop_policy == 'required':
            raise RuntimeError(f'Native Torch interop is required: {error}') from error
        self.native_interop = False
        if not self._interop_warning:
            warnings.warn(f'Native Torch interop unavailable; using CPU image upload: {error}', RuntimeWarning, stacklevel=2)
            self._interop_warning = True

    def wait_native(self):
        # SDL and Torch use different command queues. Finish previous SDL reads
        # before an external producer writes a pooled image texture/buffer.
        if self._last_fence:
            fences = (ctypes.POINTER(sdl.SDL_GPUFence) * 1)(self._last_fence)
            checked(sdl.SDL_WaitForGPUFences(self.device, True, fences, 1))

    def _prepare_native_uploads(self):
        pending = [texture for texture in self.textures.values() if texture.pending_tensor is not None]
        if pending:
            self.wait_native()
        for texture in pending:
            start = time.monotonic()
            tensor, ready = texture.pending_tensor, texture.pending_ready
            try:
                self.last_upload_backend = sdl_torch_interop.upload(texture, tensor, ready)
                texture.native_pending_copy = texture.cuda_transfer is not None
            except Exception as error:
                self.interop_failure(error)
                import torch
                if ready is not None:
                    with torch.cuda.device(tensor.device):
                        ready.synchronize()
                image = tensor.detach()
                if not image.dtype.is_floating_point:
                    image = image.to(torch.float32) / torch.iinfo(image.dtype).max
                texture.upload(image.to(device='cpu', dtype=torch.float32).numpy())
                self.last_upload_backend = 'cpu'
            texture.pending_tensor = texture.pending_ready = None
            self.last_native_upload_ms = (time.monotonic() - start) * 1000

    def _upload(self, command, draw_data):
        """Record all transfers before opening any render pass."""
        self._update_imgui_textures()
        self._prepare_native_uploads()
        transfers = []
        copy = checked(sdl.SDL_BeginGPUCopyPass(command))
        for texture in self.textures.values():
            if texture.native_pending_copy:
                source = sdl.SDL_GPUTextureTransferInfo(transfer_buffer=texture.cuda_transfer.transfer,
                    pixels_per_row=texture.width, rows_per_layer=texture.height)
                dest = sdl.SDL_GPUTextureRegion(texture=texture.texture,
                    w=texture.width, h=texture.height, d=1)
                sdl.SDL_UploadToGPUTexture(copy, ctypes.byref(source), ctypes.byref(dest), False)
                texture.native_pending_copy = False
            if texture.pending is None:
                continue
            if texture not in self._font_textures.values():
                self.last_upload_backend = 'cpu'
            transfer = self._transfer(texture.pending)
            transfers.append(transfer)
            source = sdl.SDL_GPUTextureTransferInfo(transfer_buffer=transfer,
                pixels_per_row=texture.width, rows_per_layer=texture.height)
            dest = sdl.SDL_GPUTextureRegion(texture=texture.texture,
                w=texture.width, h=texture.height, d=1)
            sdl.SDL_UploadToGPUTexture(copy, ctypes.byref(source), ctypes.byref(dest), True)
            texture.pending = None
        for name, usage, attr, stride in (
            ('vertex', sdl.SDL_GPU_BUFFERUSAGE_VERTEX, 'vtx_buffer', imgui.VERTEX_SIZE),
            ('index', sdl.SDL_GPU_BUFFERUSAGE_INDEX, 'idx_buffer', imgui.INDEX_SIZE),
        ):
            blocks = [ctypes.string_at(getattr(c, attr).data_address(), getattr(c, attr).size() * stride)
                      for c in draw_data.cmd_lists]
            data = np.frombuffer(b''.join(blocks), dtype=np.uint8)
            if not data.size:
                continue
            existing, capacity = self._buffers.get(name, (None, 0))
            if capacity < data.nbytes:
                if existing:
                    sdl.SDL_ReleaseGPUBuffer(self.device, existing)
                info = sdl.SDL_GPUBufferCreateInfo(usage=usage, size=data.nbytes + 65536)
                existing = checked(sdl.SDL_CreateGPUBuffer(self.device, ctypes.byref(info)))
                self._buffers[name] = existing, info.size
            transfer = self._transfer(data)
            transfers.append(transfer)
            source = sdl.SDL_GPUTransferBufferLocation(transfer_buffer=transfer)
            dest = sdl.SDL_GPUBufferRegion(buffer=existing, size=data.nbytes)
            sdl.SDL_UploadToGPUBuffer(copy, ctypes.byref(source), ctypes.byref(dest), True)
        sdl.SDL_EndGPUCopyPass(copy)
        return transfers

    def _begin_pass(self, command, texture, clear):
        target = sdl.SDL_GPUColorTargetInfo(texture=texture,
            clear_color=sdl.SDL_FColor(*clear), load_op=sdl.SDL_GPU_LOADOP_CLEAR,
            store_op=sdl.SDL_GPU_STOREOP_STORE)
        return checked(sdl.SDL_BeginGPURenderPass(command, ctypes.byref(target), 1, None))

    def _bind_texture(self, render_pass, texture):
        binding = sdl.SDL_GPUTextureSamplerBinding(texture=texture.texture, sampler=self.nearest_sampler if texture.nearest else self.sampler)
        sdl.SDL_BindGPUFragmentSamplers(render_pass, 0, ctypes.byref(binding), 1)

    def render_to_texture(self, command, draw_data, width, height):
        """Record the linear UI pass; exposed for offscreen/readback validation."""
        if self.scene is None or (self.scene.width, self.scene.height) != (width, height):
            if self.scene is not None:
                self.scene.release()
            self.scene = GPUTexture(self, width, height, target=True)
        transfers = self._upload(command, draw_data)
        clear = (*srgb_to_linear([0.08, 0.08, 0.08]), 1)
        render_pass = self._begin_pass(command, self.scene.texture, clear)
        sdl.SDL_BindGPUGraphicsPipeline(render_pass, self._ui_pipeline)
        if draw_data.total_vtx_count:
            binding = sdl.SDL_GPUBufferBinding(buffer=self._buffers['vertex'][0])
            sdl.SDL_BindGPUVertexBuffers(render_pass, 0, ctypes.byref(binding), 1)
            binding = sdl.SDL_GPUBufferBinding(buffer=self._buffers['index'][0])
            sdl.SDL_BindGPUIndexBuffer(render_pass, ctypes.byref(binding),
                sdl.SDL_GPU_INDEXELEMENTSIZE_16BIT if imgui.INDEX_SIZE == 2 else sdl.SDL_GPU_INDEXELEMENTSIZE_32BIT)
            sx, sy = 2 / draw_data.display_size.x, -2 / draw_data.display_size.y
            transform = np.array([sx, sy, -1 - draw_data.display_pos.x * sx,
                                  1 - draw_data.display_pos.y * sy], dtype=np.float32)
            sdl.SDL_PushGPUVertexUniformData(command, 0, transform.ctypes.data, transform.nbytes)
            index_offset = vertex_offset = 0
            fx, fy = draw_data.framebuffer_scale
            px, py = draw_data.display_pos
            for commands in draw_data.cmd_lists:
                for cmd in commands.cmd_buffer:
                    x1, y1, x2, y2 = cmd.clip_rect
                    x1, y1 = max(0, int((x1 - px) * fx)), max(0, int((y1 - py) * fy))
                    x2, y2 = min(width, int((x2 - px) * fx)), min(height, int((y2 - py) * fy))
                    if x2 <= x1 or y2 <= y1 or not cmd.elem_count:
                        continue
                    scissor = sdl.SDL_Rect(x1, y1, x2 - x1, y2 - y1)
                    sdl.SDL_SetGPUScissor(render_pass, ctypes.byref(scissor))
                    texture = self.textures[cmd.tex_ref.get_tex_id()]
                    self._bind_texture(render_pass, texture)
                    options = np.array([2 if texture.pq else int(texture.srgb), self.sdr_white_nits, 0, 0], dtype=np.float32)
                    sdl.SDL_PushGPUFragmentUniformData(command, 0, options.ctypes.data, options.nbytes)
                    sdl.SDL_DrawGPUIndexedPrimitives(render_pass, cmd.elem_count, 1,
                        cmd.idx_offset + index_offset, cmd.vtx_offset + vertex_offset, 0)
                index_offset += commands.idx_buffer.size()
                vertex_offset += commands.vtx_buffer.size()
        sdl.SDL_EndGPURenderPass(render_pass)
        return transfers

    def encode_output(self, command, destination):
        render_pass = self._begin_pass(command, destination, (0, 0, 0, 1))
        sdl.SDL_BindGPUGraphicsPipeline(render_pass, self._present_pipeline)
        self._bind_texture(render_pass, self.scene)
        options = np.array([{'sdr': 0, 'linear': 1, 'pq': 2}[self.mode],
                            self.sc_rgb_white, self.sdr_white_nits, 0], dtype=np.float32)
        sdl.SDL_PushGPUFragmentUniformData(command, 0, options.ctypes.data, options.nbytes)
        sdl.SDL_DrawGPUPrimitives(render_pass, 3, 1, 0, 0)
        sdl.SDL_EndGPURenderPass(render_pass)

    def render(self, draw_data):
        self.last_presented = False
        command = checked(sdl.SDL_AcquireGPUCommandBuffer(self.device))
        texture = ctypes.POINTER(sdl.SDL_GPUTexture)()
        width, height = ctypes.c_uint32(), ctypes.c_uint32()
        try:
            checked(sdl.SDL_WaitAndAcquireGPUSwapchainTexture(command, self.window,
                ctypes.byref(texture), ctypes.byref(width), ctypes.byref(height)))
            if not texture:
                sdl.SDL_CancelGPUCommandBuffer(command)
                return
            transfers = self.render_to_texture(command, draw_data, width.value, height.value)
            self.encode_output(command, texture)
        except BaseException:
            sdl.SDL_CancelGPUCommandBuffer(command)
            raise
        fence = checked(sdl.SDL_SubmitGPUCommandBufferAndAcquireFence(command))
        if self._last_fence:
            sdl.SDL_ReleaseGPUFence(self.device, self._last_fence)
        self._last_fence = fence
        for transfer in transfers:
            sdl.SDL_ReleaseGPUTransferBuffer(self.device, transfer)
        self.last_presented = True

    def shutdown(self):
        checked(sdl.SDL_WaitForGPUIdle(self.device))
        if self._last_fence:
            sdl.SDL_ReleaseGPUFence(self.device, self._last_fence)
            self._last_fence = None
        for tex in imgui.get_platform_io().textures:
            if tex.tex_id in self._font_textures:
                tex.set_tex_id(0)
                tex.set_status(imgui.ImTextureStatus.destroyed)
        for texture in list(self.textures.values()):
            texture.release()
        self._font_textures.clear()
        for buffer, _ in self._buffers.values():
            sdl.SDL_ReleaseGPUBuffer(self.device, buffer)
        for pipeline in (self._ui_pipeline, self._present_pipeline):
            sdl.SDL_ReleaseGPUGraphicsPipeline(self.device, pipeline)
        for shader in self._shaders:
            sdl.SDL_ReleaseGPUShader(self.device, shader)
        sdl.SDL_ReleaseGPUSampler(self.device, self.sampler)
        sdl.SDL_ReleaseGPUSampler(self.device, self.nearest_sampler)
        self.io.backend_flags &= ~(imgui.BackendFlags_.renderer_has_textures
                                   | imgui.BackendFlags_.renderer_has_vtx_offset)
        super().shutdown()
