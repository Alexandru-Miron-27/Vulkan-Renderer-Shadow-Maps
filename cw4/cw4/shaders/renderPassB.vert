#version 460

vec2 positions[3] = vec2[3](
	vec2( -1.0f, -3.0f),
	vec2( -1.0f, 1.0f),
	vec2( 3.0f, 1.0f)
);

void main()
{
	gl_Position = vec4(positions[gl_VertexIndex], 0.5, 1.0);
}

/*
layout( location = 0 ) out vec2 v2fUV;

void main()
{
	v2fUV = vec2((gl_VertexIndex << 1) & 2,
				gl_VertexIndex & 2);

	gl_Position = vec4( v2fUV * 2.0f + -1.0f, 0.0f, 1.0f );
}
*/