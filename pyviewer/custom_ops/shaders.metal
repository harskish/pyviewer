#include <metal_stdlib>
using namespace metal;

struct VertexOut {
    float4 position [[position]];
    float2 texCoord;
};

vertex VertexOut vtx_passthrough(uint vertexID [[vertex_id]]) {
    float2 pos[4] = { {-1, -1}, {1, -1}, {-1, 1}, {1, 1} };
    float2 uv[4]  = { {0, 1}, {1, 1}, {0, 0}, {1, 0} };
    VertexOut out;
    out.position = float4(pos[vertexID], 0, 1);
    out.texCoord = uv[vertexID];
    return out;
}

fragment float4 frag_copy(VertexOut in [[stage_in]],
                          texture2d<float, access::sample> src [[texture(0)]],
                          sampler s [[sampler(0)]]) {
    return src.sample(s, in.texCoord);
}