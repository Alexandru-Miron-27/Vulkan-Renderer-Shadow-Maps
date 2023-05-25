#version 460

layout(location = 0) out vec4 oColor;

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

layout(set = 1, binding = 0) uniform sampler2D baseColorTexture;
layout(set = 1, binding = 1) uniform sampler2D materialPropertiesTexture;
layout(set = 1, binding = 2) uniform sampler2D emissiveTexture;
layout(set = 1, binding = 3) uniform sampler2D normalTexture;
layout(set = 1, binding = 4) uniform sampler2D depthTexture;

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

 void computeGaussian(vec2 texcoord)
 {
 
	 /*
	vec2 tex_offset = 1.0 / textureSize(brightTexture, 0);
	vec3 result = texture(brightTexture, texcoord).rgb * weight[0];
	for(int i = 1; i < 22; i += 2)
        {
            result += texture(brightTexture, texcoord + vec2(tex_offset.x * i + tex_offset.x * 0.5, 0.0)).rgb * (weight[i] + weight[i+1]);
            result += texture(brightTexture, texcoord - vec2(tex_offset.x * i + tex_offset.x * 0.5, 0.0)).rgb * (weight[i] + weight[i+1]);
        }
		*/
 }

void main()
{
	vec2 texCoord = gl_FragCoord.xy / textureSize(baseColorTexture, 0);
	//texCoord = v2fUV;
	vec3 baseColor = texture(baseColorTexture, texCoord).rgb;
	float roughness = texture(materialPropertiesTexture, texCoord).r;
	float metalness = texture(materialPropertiesTexture, texCoord).g;
	vec3 emissiveTexture = texture(emissiveTexture, texCoord).rgb;
	vec3 normal = texture(normalTexture, texCoord).rgb;
	float depth = texture(depthTexture, texCoord).r;

	// see how to get vertex position in WCS
	vec2 ndc; //normalized device coordinates
	ndc.x = gl_FragCoord.x / textureSize(baseColorTexture, 0).x * 2.0 - 1.0;
	ndc.y = gl_FragCoord.y / textureSize(baseColorTexture, 0).y * 2.0 - 1.0;

	//clip-space position 
	vec4 clipSpacePosition = vec4(ndc, depth*2.0 - 1.0, 1.0);

	mat4 inverseProjection = uScene.inverseProjection;
	mat4 inverseCamera = uScene.inverseCamera;

	// transform from clip space to view space
	vec4 viewSpacePos = inverseProjection * clipSpacePosition;
	viewSpacePos /= viewSpacePos.w;

	// transform from view space to world space
	vec4 worldSpacePos = inverseCamera * viewSpacePos;
	worldSpacePos /= worldSpacePos.w;

	vec3 viewDir = normalize(uScene.cameraPos - worldSpacePos.xyz);
	vec3 lightDir = normalize(uScene.lightPosition - worldSpacePos.xyz);
	vec3 halfVec = normalize(viewDir + lightDir);

	//compute lighting
	//calculating shininess
	float shininess = 2.0 / (pow(roughness, 4.0) + eps) - 2.0;
	//Lambertian diffuse
	vec3 F0 = ( 1.0 - metalness) * vec3(0.04)  + metalness * baseColor;
	vec3 F = F0 + (1.0 - F0) * pow(1.0 - dot(halfVec, viewDir), 5.0);
	vec3 lambertianDiffuse = (baseColor / pi) * (vec3(1.0) - F)*(1.0 - metalness); 

	//normal distribution function D
	float D = computeD(normal, halfVec, shininess);

	float G = computeG(normal, viewDir, lightDir, halfVec, roughness);

	// assume ambient illumination
	vec3 ambientTerm = baseColor * vec3(0.02);

	vec3 fr = lambertianDiffuse + (D * F * G) /
	4 * max(dot(normal, viewDir), 0.0) * max(dot(normal, lightDir), 0.0);

	//emitted light is 0
	vec3 Lo = emissiveTexture + ambientTerm + fr * uScene.lightColor
	* max(dot(normal, lightDir), 0.0);

	//Tone mapping
	Lo = Lo/ (1.0 + Lo);

	//BLOOM
	vec4 brightColor;
	if (max(max(Lo.x, Lo.y), Lo.z) > 1.0)
	{
		brightColor = vec4(Lo, 1.0);
		//computeGaussian(texCoord);
	}
	else
	{
		brightColor = vec4(0.0, 0.0, 0.0, 1.0);
	}

	


	oColor = vec4(Lo, 1.0);

}