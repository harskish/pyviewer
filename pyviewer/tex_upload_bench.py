import numpy as np
import glfw
glfw.ERROR_REPORTING = 'raise' # make sure errors don't get swallowed
import OpenGL.GL as gl
import time

def test(
    W,
    H,
    internal_fmt=gl.GL_RGB32F,  # how OGL stores data
    incoming_fmt=gl.GL_RGB,     # incoming channel format
    incoming_dtype=gl.GL_FLOAT, # incoming dtype
    mipmap=False,
    reallocate=False,           # glTexSubImage2D or glTexImage2D?
    unpack_alignment=4,
    upload_block_size=-1,       # upload in blocks (forum.beyond3d.com/threads/texture-upload-speed-issue-opengl.45464/post-1289545)
    verbose=False,
):
    # internalformat
    # Specifies the number of color components in the texture
    assert internal_fmt in [
        # Table 1: Base formats
        gl.GL_DEPTH_COMPONENT, gl.GL_DEPTH_STENCIL, gl.GL_RED, gl.GL_RG, gl.GL_RGB, gl.GL_RGBA,
        # Table 2: Sized Internal Formats 
        gl.GL_R8, gl.GL_R8_SNORM, gl.GL_R16, gl.GL_R16_SNORM, gl.GL_RG8, gl.GL_RG8_SNORM, gl.GL_RG16, gl.GL_RG16_SNORM, gl.GL_R3_G3_B2, gl.GL_RGB4, gl.GL_RGB5, gl.GL_RGB8, gl.GL_RGB8_SNORM, gl.GL_RGB10, gl.GL_RGB12, gl.GL_RGB16_SNORM, gl.GL_RGBA2, gl.GL_RGBA4, gl.GL_RGB5_A1, gl.GL_RGBA8, gl.GL_RGBA8_SNORM, gl.GL_RGB10_A2, gl.GL_RGB10_A2UI, gl.GL_RGBA12, gl.GL_RGBA16, gl.GL_SRGB8, gl.GL_SRGB8_ALPHA8, gl.GL_R16F, gl.GL_RG16F, gl.GL_RGB16F, gl.GL_RGBA16F, gl.GL_R32F, gl.GL_RG32F, gl.GL_RGB32F, gl.GL_RGBA32F, gl.GL_R11F_G11F_B10F, gl.GL_RGB9_E5, gl.GL_R8I, gl.GL_R8UI, gl.GL_R16I, gl.GL_R16UI, gl.GL_R32I, gl.GL_R32UI, gl.GL_RG8I, gl.GL_RG8UI, gl.GL_RG16I, gl.GL_RG16UI, gl.GL_RG32I, gl.GL_RG32UI, gl.GL_RGB8I, gl.GL_RGB8UI, gl.GL_RGB16I, gl.GL_RGB16UI, gl.GL_RGB32I, gl.GL_RGB32UI, gl.GL_RGBA8I, gl.GL_RGBA8UI, gl.GL_RGBA16I, gl.GL_RGBA16UI, gl.GL_RGBA32I, gl.GL_RGBA32UI,
        # Table 3: Compressed Internal Formats
        gl.GL_COMPRESSED_RED, gl.GL_COMPRESSED_RG, gl.GL_COMPRESSED_RGB, gl.GL_COMPRESSED_RGBA, gl.GL_COMPRESSED_SRGB, gl.GL_COMPRESSED_SRGB_ALPHA, gl.GL_COMPRESSED_RED_RGTC1, gl.GL_COMPRESSED_SIGNED_RED_RGTC1, gl.GL_COMPRESSED_RG_RGTC2, gl.GL_COMPRESSED_SIGNED_RG_RGTC2, gl.GL_COMPRESSED_RGBA_BPTC_UNORM, gl.GL_COMPRESSED_SRGB_ALPHA_BPTC_UNORM, gl.GL_COMPRESSED_RGB_BPTC_SIGNED_FLOAT, gl.GL_COMPRESSED_RGB_BPTC_UNSIGNED_FLOAT,
    ], 'Invalid internal format'

    # format
    # Specifies the format of the pixel data
    assert incoming_fmt in [
        gl.GL_RED, gl.GL_RG, gl.GL_RGB, gl.GL_BGR, gl.GL_RGBA, gl.GL_BGRA, gl.GL_RED_INTEGER, gl.GL_RG_INTEGER, gl.GL_RGB_INTEGER, gl.GL_BGR_INTEGER, gl.GL_RGBA_INTEGER, gl.GL_BGRA_INTEGER, gl.GL_STENCIL_INDEX, gl.GL_DEPTH_COMPONENT, gl.GL_DEPTH_STENCIL
    ], 'Invalid incoming pixel format'

    # type
    # Specifies the data type of the pixel data
    assert incoming_dtype in [
        gl.GL_UNSIGNED_BYTE, gl.GL_BYTE, gl.GL_UNSIGNED_SHORT, gl.GL_SHORT, gl.GL_UNSIGNED_INT, gl.GL_INT, gl.GL_HALF_FLOAT, gl.GL_FLOAT, gl.GL_UNSIGNED_BYTE_3_3_2, gl.GL_UNSIGNED_BYTE_2_3_3_REV, gl.GL_UNSIGNED_SHORT_5_6_5, gl.GL_UNSIGNED_SHORT_5_6_5_REV, gl.GL_UNSIGNED_SHORT_4_4_4_4, gl.GL_UNSIGNED_SHORT_4_4_4_4_REV, gl.GL_UNSIGNED_SHORT_5_5_5_1, gl.GL_UNSIGNED_SHORT_1_5_5_5_REV, gl.GL_UNSIGNED_INT_8_8_8_8, gl.GL_UNSIGNED_INT_8_8_8_8_REV, gl.GL_UNSIGNED_INT_10_10_10_2, gl.GL_UNSIGNED_INT_2_10_10_10_REV
    ], 'Invalid incoming pixel dtype'
    
    np_dtype = {
        gl.GL_FLOAT: np.float32,
        gl.GL_HALF_FLOAT: np.float16,
        gl.GL_UNSIGNED_BYTE: np.uint8,
        gl.GL_UNSIGNED_SHORT: np.uint16,
    }[incoming_dtype]
    C = 4 if incoming_fmt == gl.GL_RGBA else 3
    
    texture_data = np.random.rand(H, W, C)
    if not np.issubdtype(np_dtype, np.floating):
        texture_data = (texture_data * 255).clip(0, 255)
    texture_data = texture_data.astype(np_dtype).copy()

    # RGB UINT8 data: no guarantee of 4-byte row alignment
    # (F32 or RGBA alignment always divisible by 4)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, unpack_alignment) # default: 4 bytes

    # Create texture
    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR if mipmap else gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR if mipmap else gl.GL_LINEAR)
    if mipmap:
        gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
    
    gl.glFinish()

    data = texture_data                  # 83ms
    #data = texture_data.tobytes()       # 86ms
    #data = bytearray(texture_data)      # 86ms
    #data = bytearray(texture_data.data) # 83ms
    #data = texture_data.data            # 84ms

    timings_ns = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1:
        t_start = time.monotonic_ns()

        if reallocate or not timings_ns: # first call
            # (Re)allocate
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, # GLenum target
                0,                # GLint level
                internal_fmt,     # GLint internalformat
                W,                # GLsizei width
                H,                # GLsizei height
                0,                # GLint border
                incoming_fmt,     # GLenum format
                incoming_dtype,   # GLenum type
                data,             # const void * data
            )
        else:
            # Overwrite
            block = upload_block_size if upload_block_size > 0 else H
            for y in range(0, H, block):
                block_height = min(block, H - y)
                gl.glTexSubImage2D(
                    gl.GL_TEXTURE_2D,  # GLenum target
                    0,                 # GLint level
                    0,                 # GLint xoffset
                    y,                 # GLint yoffset
                    W,                 # GLsizei width
                    block_height,      # GLsizei height
                    incoming_fmt,      # GLenum format
                    incoming_dtype,    # GLenum type
                    data[y:y+block_height, :, :],  # Block of data
                )
        
        if mipmap:
            gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
        
        gl.glFinish()
        dt_ns = time.monotonic_ns() - t_start
        timings_ns.append(dt_ns)
    
    # Theoretical maximum speed if bandwidth bound
    # Apple M1 Pro:
    bytes_per_s = 200 * 1e9 # 200GB/s
    bytes_image = texture_data.nbytes
    min_time_m1 = 1000 * 2 * bytes_image / bytes_per_s
    
    ms_per_img = np.mean(timings_ns) / 1e6
    if verbose:
        print(f'[{upload_block_size}x{W}] dt={ms_per_img:.3f}ms (M1 Pro ideal: {min_time_m1:.3f}ms)')

    # Clean up
    gl.glDeleteTextures([tex])

    return ms_per_img


if __name__ == '__main__':
    # Benchmark texture upload speed

    # Initialize GLFW
    if not glfw.init():
        raise RuntimeError("Failed to initialize GLFW")

    # Create a hidden window (no need to display it for benchmarking)
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    window = glfw.create_window(640, 480, "Texture Upload Benchmark", None, None)
    if not window:
        glfw.terminate()
        raise RuntimeError("Failed to create GLFW window")

    glfw.make_context_current(window)
    
    for in_dtype in ['GL_UNSIGNED_BYTE', 'GL_FLOAT', 'GL_HALF_FLOAT']:
        for in_fmt in ['GL_RGB', 'GL_RGBA']:
            for W, H in [(256, 256), (512, 320), (512, 512),
                        (800, 600), (1280, 720), (1920, 1080),
                        (2000, 2000), (4096, 4096), (6000, 4000)]:
                for unpack in [1, 4, 8]:
                    timings = []
                    block_sizes = set([H, 512, 320, 256, 240, 196, 128, 96, 80, 64, 40, 32])
                    for block_sz in [b for b in sorted(block_sizes, reverse=True) if b <= H]:
                        storage_fmt = gl.GL_RGB
                        dt_ms = test(W, H, storage_fmt, getattr(gl, in_fmt), getattr(gl, in_dtype), reallocate=False, unpack_alignment=unpack, upload_block_size=block_sz)
                        timings.append((dt_ms.item(), block_sz))
                
                    print(f'{in_dtype}, {in_fmt}, unpack={unpack}, res={W}x{H}: ', end='')
                    print(', '.join([f'({B}, {dt:.3f})' for dt,B in sorted(timings)[:3]]))

    # Clean up GLFW
    glfw.terminate()