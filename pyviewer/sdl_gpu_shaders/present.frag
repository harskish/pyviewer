#version 450
layout(location=0) in vec2 frag_uv;
layout(set=2, binding=0) uniform sampler2D scene;
// mode: 0=SDR sRGB, 1=extended linear scRGB, 2=BT.2020 PQ.
// SDR white is expressed in scene-relative scRGB units and in absolute nits.
layout(set=3, binding=0) uniform OutputInfo { vec4 options; } info;
layout(location=0) out vec4 out_color;
vec3 linear_to_srgb(vec3 c) {
    return mix(c * 12.92, 1.055 * pow(max(c, vec3(0)), vec3(1.0/2.4)) - 0.055, greaterThan(c, vec3(0.0031308)));
}
vec3 pq_encode(vec3 nits) {
    vec3 p = pow(clamp(nits / 10000.0, 0, 1), vec3(2610.0 / 16384.0));
    return pow((3424.0/4096.0 + (2413.0/128.0)*p) / (1 + (2392.0/128.0)*p), vec3(2523.0/32.0));
}
void main() {
    vec3 rgb = texture(scene, frag_uv).rgb;
    if (info.options.x < 0.5) rgb = linear_to_srgb(rgb);
    else if (info.options.x < 1.5) rgb *= info.options.y;
    else {
        // Linear Rec.709 -> linear Rec.2020 (matrix columns).
        rgb = mat3(0.627404, 0.069097, 0.016391,
                   0.329283, 0.919540, 0.088013,
                   0.043313, 0.011362, 0.895595) * rgb;
        rgb = pq_encode(rgb * info.options.z);
    }
    out_color = vec4(rgb, 1);
}
