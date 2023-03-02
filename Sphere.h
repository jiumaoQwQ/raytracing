#pragma once

struct Sphere
{
    float3 pos;
    float radius;
    float3 color;
    int material;
    Sphere(float3 _pos, float _radius, int _material, float3 _color) : pos(_pos), radius(_radius), material(_material), color(_color) {}
};

struct Ray
{
    float3 pos;
    float3 dir;
};
