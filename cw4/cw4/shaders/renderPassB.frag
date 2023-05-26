#version 460

layout (location = 0) in vec3 vsPos;
layout (location = 1) in vec3 vsNormal;
layout (location = 2) in vec2 vsTexCoord;
layout (location = 3) in vec4 vsTangent;
layout (location = 4) in vec3 vsBitangent;

layout(location = 0) out vec4 oColor;

layout( set = 0, binding = 0) uniform UScene
 {
	mat4 camera;
	mat4 projection; 
	mat4 projCam;
	vec3 cameraPos;

	vec3 lightPosition;
	vec3 lightColor;
 } uScene;

layout (set = 1, binding = 0) uniform sampler2D uTexBaseColor;
layout (set = 1, binding = 1) uniform sampler2D uTexRoughnessColor;
layout (set = 1, binding = 2) uniform sampler2D uTexMetalColor;
layout (set = 1, binding = 3) uniform sampler2D uNormalMap;

 const float eps = 0.0001;
 const float pi = 3.141592;

 float weight[22];

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

	vec3 baseColor = texture(uTexBaseColor, vsTexCoord).rgb;
	float roughness = texture(uTexRoughnessColor, vsTexCoord).r;
	float metalness = texture(uTexMetalColor, vsTexCoord).r;
	vec3 normalMap = normalize((texture(uNormalMap, vsTexCoord).rgb) * 2.0 - 1.0);

	//TBN matrix
	mat3 TBN = mat3(vsTangent.xyz, vsBitangent, vsNormal);
	normal = normalize(TBN * normalMap);

	//calculating shininess
	float shininess = (2.0 / (pow(roughness, 4.0) + eps)) - 2.0;

	//Lambertian diffuse
	vec3 F0 = ( 1.0 - metalness) * vec3(0.04)  + metalness * baseColor;
	vec3 F = F0 + (1.0 - F0) * pow(1.0 - dot(halfVec, viewDir), 5.0);
	vec3 lambertianDiffuse = (baseColor / pi) * (vec3(1.0) - F)*(1.0 - metalness); 

	//normal distribution function D
	float D = computeD(normal, halfVec, shininess);

	float G = computeG(normal, viewDir, lightDir, halfVec, roughness);

	// assume ambient illumination
	vec3 ambientTerm = baseColor * vec3(0.02);

	vec3 pbrSpecular = (D * F * G) /
	(4.0 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0));
	vec3 fr = lambertianDiffuse + pbrSpecular;

	//emitted light is 0
	vec3 Lo = 0.0 + ambientTerm + fr * uScene.lightColor
	* max(dot(normal, lightDir), 0.0);

	oColor = vec4(baseColor, 1.0);

}