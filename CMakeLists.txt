cmake_minimum_required(VERSION 3.20)

project(raytracing LANGUAGES CXX CUDA)

add_executable(main main.cu Camera.h MeshVao.h Texture.h)

add_subdirectory(external)

target_link_libraries(main glm glfw glad)

add_custom_command(
        TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/shader
                ${CMAKE_CURRENT_BINARY_DIR}/Debug/shader)

add_custom_command(
        TARGET main POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
                ${CMAKE_SOURCE_DIR}/shader
                ${CMAKE_CURRENT_BINARY_DIR}/Release/shader)