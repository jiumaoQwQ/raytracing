#pragma once

#include "helper_math.h"
#include "Scene.h"

enum MoveDirection
{
    FORWARD,
    BACKWARD,
    LEFT,
    RIGHT
};

struct Camera
{
    float3 pos;
    float3 lookAtPos;
    float3 up;
    float fov;
    float distance;
    float ratio;

    __device__ void init()
    {
        pos = {0.0f, 1.0f, -5.0f};
        lookAtPos = {0.0f, 1.0f, -1.0f};
        up = {0.0f, 1.0f, 0.0f};
        fov = 1.0f / 3.0f * PI;
        distance = 1.0f;
        ratio = 1920.0f / 1080.0f;
    }

    __device__ float getHeight()
    {
        return 2.0f * __tanf(fov / 2.0f) * distance;
    }
    __device__ float getWidth()
    {
        return getHeight() * ratio;
    }

    __device__ float3 getW()
    {
        return normalize(pos - lookAtPos);
    }
    __device__ float3 getU()
    {
        return normalize(cross(up, getW()));
    }
    __device__ float3 getV()
    {
        return normalize(cross(getW(), getU()));
    }
    __device__ float3 getLeftButtomPos()
    {
        return pos - getW() * distance - getU() * getWidth() / 2.0f - getV() * getHeight() / 2.0f;
    }

    __device__ Ray getRay(float u, float v)
    {
        Ray ray;
        ray.pos = pos;
        float3 pixel_pos = getLeftButtomPos() + getU() * u * getWidth() + getV() * v * getHeight();
        ray.dir = normalize(pixel_pos - pos);
        return ray;
    }

    __device__ void move(MoveDirection dir)
    {
        float speed = 0.1f;
        if (dir == FORWARD)
        {
            pos += -getW() * speed;
        }
        else if (dir == BACKWARD)
        {
            pos += getW() * speed;
        }
        else if (dir == RIGHT)
        {
            pos += getU() * speed;
        }
        else if (dir == LEFT)
        {
            pos += -getU() * speed;
        }
    }
};