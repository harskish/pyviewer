#version 450
layout(location=0) out vec2 frag_uv;
void main() {
    vec2 p = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
    frag_uv = p;
    gl_Position = vec4(p.x * 2 - 1, 1 - p.y * 2, 0, 1);
}
