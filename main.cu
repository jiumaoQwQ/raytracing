#define PI 3.14159265
#define WIDTH 1920
#define HEIGHT 1080
#define sample_per_pixel 4
#define num_of_spheres 10

#include <iostream>
#include <vector>
#include "Camera.h"

#include <glad/gl.h>
#include <GLFW/glfw3.h>
#include "MeshVao.h"
#include "Shader.h"
#include "Texture.h"
#include "Scene.h"
#include <curand_kernel.h>

GLFWwindow *window;
Sphere *spheres;

__host__ void init_glfw()
{
    glfwInit();

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(1920, 1080, "RayTracing", NULL, NULL);
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow *window, int width, int height)
                                   { glViewport(0, 0, width, height); });
    gladLoadGL(glfwGetProcAddress);
}

__global__ void init_kernel(float3 *colors, curandState *state)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > WIDTH || y > HEIGHT)
        return;
    int id = x + y * WIDTH;

    colors[id] = {0, 0, 0};
    curand_init(1234, id, 0, &state[id]);
}

__global__ void one_thread_init(Camera *camera)
{
    camera->init();
}

__device__ float3 sampleInSphereSurface(curandState *state)
{
    float randomf1 = curand_uniform(state);
    float randomf2 = curand_uniform(state);
    float phi = randomf1 * 2 * 3.1415926;
    float theta = acosf((randomf2 - 0.5) * 2);

    return {__cosf(phi) * __sinf(theta),
            __sinf(phi) * __sinf(theta),
            __cosf(theta)};
}

__device__ void test_if_intersect(Sphere *spheres, Ray ray,
                                  bool &is_hit, int &index,
                                  float3 &hit_pos, float3 &hit_normal)
{
    float3 O = ray.pos;
    float3 d = ray.dir;
    float t = 10000000000;
    float tmpt = 0;
    is_hit = false;
    index = -1;
    for (int i = 0; i < num_of_spheres; i++)
    {
        float r = spheres[i].radius;
        float3 C = spheres[i].pos;
        float3 o_minus_c = O - C;
        float a = dot(d, d);
        float b = 2 * dot(d, o_minus_c);
        float c = dot(o_minus_c, o_minus_c) - r * r;
        float delta = b * b - 4 * a * c;
        if (delta > 0)
        {
            float t1 = (-b + sqrtf(delta)) / (2 * a);
            float t2 = (-b - sqrtf(delta)) / (2 * a);
            if (t1 > 0 && t2 > 0)
            {
                tmpt = __min(t1, t2);
            }
            else if (t1 > 0 && t2 < 0)
            {
                tmpt = t1;
            }
            else
            {
                continue;
            }
            if (tmpt < t)
            {
                is_hit = true;
                t = tmpt;
                index = i;
            }
        }
    }
    if (is_hit)
    {
        hit_pos = O + d * t;
        hit_normal = normalize(hit_pos - spheres[index].pos);
    }
}

__device__ float3 cast_ray(Sphere *spheres, Ray ray, curandState *state)
{
    float3 m_color = {0, 0, 0};
    float3 brightness = {1, 1, 1};
    Ray m_ray = ray;
    for (int i = 0; i < 10; i++)
    {
        int sphere_index;
        bool is_hit;
        float3 hit_pos, hit_normal;
        test_if_intersect(spheres, m_ray, is_hit, sphere_index, hit_pos, hit_normal);
        if (is_hit)
        {
            int material = spheres[sphere_index].material;
            if (material == 0)
            {
                m_color = brightness * spheres[sphere_index].color;
                break;
            }
            else if (material == 1)
            {
                brightness *= spheres[sphere_index].color;
                float3 random_vec = sampleInSphereSurface(state);
                m_ray.pos = hit_pos;
                m_ray.dir = normalize(hit_pos + hit_normal + random_vec - hit_pos);
                continue;
            }
            else if (material == 2 || material == 4)
            {
                float fuzz = 0;
                if (material == 4)
                    fuzz = 0.4;
                m_ray.pos = hit_pos;
                m_ray.dir = m_ray.dir - 2 * dot(hit_normal, m_ray.dir) * hit_normal;
                float3 random_vec = sampleInSphereSurface(state);
                m_ray.dir += random_vec * fuzz;
                if (dot(m_ray.dir, hit_normal) < 0)
                    break;
                else
                    brightness *= spheres[sphere_index].color;
                continue;
            }
            else if (material == 3)
            {
            }
        }
        else
        {
            m_color = make_float3(0, 0, 0) * brightness;
            break;
        }
    }
    return m_color;
}

__global__ void substep_kernel(float3 *colors, Sphere *spheres, Camera *camera, curandState *state)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > WIDTH || y > HEIGHT)
        return;
    int id = x + y * WIDTH;

    float3 m_color = {0, 0, 0};

    float u = (x + curand_uniform(&state[id])) * 1.0f / WIDTH;
    float v = (y + curand_uniform(&state[id])) * 1.0f / HEIGHT;

    for (int i = 0; i < sample_per_pixel; i++)
    {
        auto ray = camera->getRay(u, v);
        m_color += cast_ray(spheres, ray, state);
    }
    m_color /= sample_per_pixel;
    colors[id] += m_color;
}

__global__ void transfor_to_canvas(float3 *canvas, float3 *colors, int cnt)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if (x > WIDTH || y > HEIGHT)
        return;
    int id = x + y * WIDTH;

    canvas[id] = colors[id] * 1.0 / cnt;
}

// void process_input(GLFWwindow* window)
// {
//     if(glfwGetMouseButton(window,GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS){

//     }
// }

int main()
{
    init_glfw();
    Shader shader("./shader/shader.vs", "./shader/shader.fs");
    shader.use();
    Texture texture;

    MeshVao square_vao(4, 6);
    {
        std::vector<float3> pos = {{1, 1, 0}, {1, -1, 0}, {-1, -1, 0}, {-1, 1, 0}};
        std::vector<float2> tex = {{1, 1}, {1, 0}, {0, 0}, {0, 1}};
        std::vector<unsigned int> index = {0, 1, 2, 0, 2, 3};
        square_vao.copyIn(pos.data(), nullptr, tex.data(), index.data(),
                          pos.size() * sizeof(float3), 0, tex.size() * sizeof(float2), index.size() * sizeof(unsigned int));
    }

    {
        std::vector<Sphere> cpu_spheres;
        cpu_spheres.push_back(Sphere({0, 5.4, -1}, 3.0f, 0, {10.0f, 10.0f, 10.0f}));
        cpu_spheres.push_back(Sphere({0, -100.5, -1}, 100.0f, 1, {0.8, 0.8, 0.8}));
        cpu_spheres.push_back(Sphere({0, 102.5, -1}, 100.0f, 1, {0.8, 0.8, 0.8}));
        cpu_spheres.push_back(Sphere({0, 1, 101}, 100.0f, 1, {0.8, 0.8, 0.8}));
        cpu_spheres.push_back(Sphere({-101.5, 0, -1}, 100.0f, 1, {0.6, 0.0, 0.0}));
        cpu_spheres.push_back(Sphere({101.5, 0, -1}, 100.0f, 1, {0.0, 0.6, 0.0}));

        cpu_spheres.push_back(Sphere({0, -0.2, -1.5}, 0.3f, 1, {0.8, 0.3, 0.3}));
        cpu_spheres.push_back(Sphere({-0.8, 0.2, -1}, 0.7f, 2, {0.6, 0.8, 0.8}));
        cpu_spheres.push_back(Sphere({0.7, 0, -0.5}, 0.5f, 3, {1.0, 1.0, 1.0}));
        cpu_spheres.push_back(Sphere({0.6, -0.3, -2.0}, 0.2f, 4, {0.8, 0.6, 0.2}));
        cudaMalloc(&spheres, num_of_spheres * sizeof(Sphere));
        cudaMemcpy(spheres, cpu_spheres.data(), num_of_spheres * sizeof(Sphere), cudaMemcpyHostToDevice);
    }

    Camera *camera;
    float3 *canvas;
    float3 *colors;
    float3 *cpu_colors = (float3 *)malloc(WIDTH * HEIGHT * sizeof(float3));
    curandState *state;

    cudaMalloc(&camera, sizeof(camera));
    cudaMalloc(&canvas, WIDTH * HEIGHT * sizeof(float3));
    cudaMalloc(&colors, WIDTH * HEIGHT * sizeof(float3));
    cudaMalloc(&state, WIDTH * HEIGHT * sizeof(curandState));

    one_thread_init<<<1, 1>>>(camera);
    init_kernel<<<{(WIDTH + 7) / 8, (HEIGHT + 7) / 8}, {8, 8}>>>(colors, state);

    int cnt = 0;
    while (!glfwWindowShouldClose(window))
    {
        glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        // process_input(window);
        cnt++;
        substep_kernel<<<{(WIDTH + 7) / 8, (HEIGHT + 7) / 8}, {8, 8}>>>(colors, spheres, camera, state);

        transfor_to_canvas<<<{(WIDTH + 7) / 8, (HEIGHT + 7) / 8}, {8, 8}>>>(canvas, colors, cnt);
        cudaMemcpy(cpu_colors, canvas, WIDTH * HEIGHT * sizeof(float3), cudaMemcpyDeviceToHost);
        texture.copyIn(WIDTH, HEIGHT, cpu_colors);
        texture.bind(0);

        square_vao.draw();

        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    cudaFree(camera);
    cudaFree(canvas);
    cudaFree(colors);
    cudaFree(state);
    free(cpu_colors);
}