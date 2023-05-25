#version 450

layout (location = 0) in vec3 vsPos;
layout (location = 1) in vec3 vsNormal;
layout (location = 2) in vec2 vsTexCoord;
layout (location = 3) in vec4 vsTangent;
layout (location = 4) in vec3 vsBitangent;

layout( set = 0, binding = 0) uniform UScene
 {
	mat4 camera;
	mat4 projection; 
	mat4 projCam;
	mat4 inverseProjection;
	mat4 inverseCamera;
	vec3 cameraPos;

	vec3 lightPosition;
	vec3 lightColor;
 } uScene;

layout (set = 1, binding = 0) uniform sampler2D uTexBaseColor;
layout (set = 1, binding = 1) uniform sampler2D uTexRoughnessColor;
layout (set = 1, binding = 2) uniform sampler2D uTexMetalColor;
layout (set = 1, binding = 3) uniform sampler2D uNormalMap;

layout( set = 2, binding = 0) uniform MaterialUBO
 {
	vec3 baseColor;
	float roughness;
	float metalness;
	vec3 emissiveColor;
 } material;


 layout (location = 0) out vec4 oBaseColor;
 layout (location = 1) out vec2 oMaterialProps;
 layout (location = 2) out vec4 oEmissive;
 layout (location = 3) out vec4 oNormal;

 const float eps = 0.0001;
 const float pi = 3.141592;

 float computeD( vec3 normal, vec3 halfVec, float shininess)
 {
	// using the Blinn-Phong distribution - similar to specular
	float nh = max(dot(normal, halfVec), 0.0);
	float D = (shininess + 2.0) * pow(nh, shininess) /
	(2.0 * pi);
	return D;
 }

 float computeG(vec3 normal, vec3 viewDir, vec3 lightDir, vec3 halfVec, float roughness)
 {
	float nhClamped = max(dot(normal, halfVec), 0.0);
	float nvClamped = max(dot(normal, viewDir), 0.0); 
	float nlClamped = max(dot(normal, lightDir), 0.0);

	float A = 2 * nhClamped * nvClamped / dot(viewDir, halfVec);
	float B = 2 * nhClamped * nlClamped / dot(viewDir, halfVec);
	float G = min(min(A,B),1.0);
	return G;
 }


void main()
{
	vec3 viewDir = normalize(uScene.cameraPos - vsPos);
	vec3 lightDir = normalize(uScene.lightPosition - vsPos);
	vec3 halfVec = normalize(viewDir + lightDir);
	vec3 normal = normalize(vsNormal);

	vec3 baseColor = texture(uTexBaseColor, vsTexCoord).rgb * material.baseColor;
	float roughness = texture(uTexRoughnessColor, vsTexCoord).r * material.roughness;
	float metalness = texture(uTexMetalColor, vsTexCoord).r * material.metalness;
	
	oBaseColor = vec4(baseColor, 1.0);
	oMaterialProps = vec2(roughness, metalness);
	oEmissive = vec4(material.emissiveColor, 1.0);
	oNormal = vec4(normal, 0.0);
	
}
