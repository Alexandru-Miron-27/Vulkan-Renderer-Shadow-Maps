#version 450
 
layout (location = 0) in vec3 iPosition;
layout (location = 1) in vec3 iNormal;
layout (location = 2) in vec2 iTexCoord;
layout (location = 3) in vec4 iTangent;

layout( set = 0, binding = 0) uniform UScene
 {
	mat4 camera;
	mat4 projection; 
	mat4 projCam;
	vec3 cameraPos;

	vec3 lightPosition;
	vec3 lightColor;
 } uScene;


layout (location = 0) out vec3 vsPos;
layout (location = 1) out vec3 vsNormal;
layout (location = 2) out vec2 vsTexCoord;
layout (location = 3) out vec4 vsTangent;
layout (location = 4) out vec3 vsBitangent;


void main()
{
	vsPos = iPosition;
	vsTexCoord = iTexCoord;
	vsNormal = iNormal;
	vsTangent = iTangent;
	//computing bitangent from tangent
	vsBitangent = cross(iNormal, iTangent.xyz) * iTangent.w;
	gl_Position = uScene.projCam * vec4( iPosition, 1.f);
}
