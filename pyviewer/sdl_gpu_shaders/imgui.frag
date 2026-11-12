#version 450
layout(location=0) in vec2 frag_uv;
layout(location=1) in vec4 frag_color;
layout(set=2, binding=0) uniform sampler2D image_texture;
layout(set=3, binding=0) uniform TextureInfo { vec4 options; } info;
layout(location=0) out vec4 out_color;
vec3 srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow(max((c + 0.055) / 1.055, vec3(0)), vec3(2.4)), greaterThan(c, vec3(0.04045)));
}
vec3 pq_to_linear(vec3 c) {
    vec3 p = pow(clamp(c, 0, 1), vec3(32.0 / 2523.0));
    return 10000.0 * pow(max(p - 3424.0/4096.0, vec3(0)) /
        max(2413.0/128.0 - (2392.0/128.0)*p, vec3(1e-6)), vec3(16384.0/2610.0));
}
void main() {
    vec4 texel = texture(image_texture, frag_uv);
    if (info.options.x > 1.5) {
        // Rec.2020/PQ -> relative linear Rec.709. Keep negative components
        // so wide-gamut colors survive the final Rec.709 -> Rec.2020 pass.
        texel.rgb = mat3(1.660491, -0.124550, -0.018151,
                       -0.587641, 1.132900, -0.100579,
                       -0.072850, -0.008349, 1.118730)
                    * pq_to_linear(texel.rgb) / info.options.y;
    } else if (info.options.x > 0.5) texel.rgb = srgb_to_linear(texel.rgb);
    out_color = frag_color * texel;
}
