#version 450
 
layout (location = 0) in vec3 iPosition;

layout( set = 0, binding = 0) uniform UScene
 {
	mat4 camera;
	mat4 projection; 
	mat4 projCam;
	mat4 lightSpaceMatrix;
	vec3 cameraPos;

	vec3 lightPosition;
	vec3 lightColor;
 } uScene;



void main()
{
	gl_Position = uScene.lightSpaceMatrix * vec4( iPosition, 1.f);
}
