#version 450
layout(location=0) in vec2 position;
layout(location=1) in vec2 uv;
layout(location=2) in vec4 color;
layout(set=1, binding=0) uniform Transform { vec2 scale; vec2 translate; } transform;
layout(location=0) out vec2 frag_uv;
layout(location=1) out vec4 frag_color;
vec3 srgb_to_linear(vec3 c) {
    return mix(c / 12.92, pow((c + 0.055) / 1.055, vec3(2.4)), greaterThan(c, vec3(0.04045)));
}
void main() {
    frag_uv = uv;
    frag_color = vec4(srgb_to_linear(color.rgb), color.a);
    gl_Position = vec4(position * transform.scale + transform.translate, 0, 1);
}
