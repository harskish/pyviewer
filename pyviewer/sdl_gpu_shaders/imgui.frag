#version 450
layout(location=0) in vec2 frag_uv;
layout(location=1) in vec4 frag_color;
layout(set=2, binding=0) uniform sampler2D image_texture;
layout(set=3, binding=0) uniform TextureInfo { vec4 options; } info;
layout(location=0) out vec4 out_color;
vec3 srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow(max((c + 0.055) / 1.055, vec3(0)), vec3(2.4)), greaterThan(c, vec3(0.04045)));
}
void main() {
    vec4 texel = texture(image_texture, frag_uv);
    if (info.options.x > 0.5) texel.rgb = srgb_to_linear(texel.rgb);
    out_color = frag_color * texel;
}
