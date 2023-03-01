#version 330 core
in vec2 vTex;

out vec4 FragColor;  

uniform sampler2D texture1;

void main()
{
    vec3 color = texture(texture1,vTex).rgb;
    vec3 mapped = color / (color+vec3(1));
    mapped = pow(mapped,vec3(1/2.2));
    FragColor = vec4(mapped,1.0);
}