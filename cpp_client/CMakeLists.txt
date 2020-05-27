cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Produce compile_commands.json
project(custom_ops)
set(LIBRARY_TARGET_NAME ${PROJECT_NAME})
find_package(Torch REQUIRED)

#set(${LIBRARY_TARGET_NAME}_SRC
    #NMS.cpp
#)

#set(${LIBRARY_TARGET_NAME}_HDR
    #NMS.hpp
#)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g3 -O3")
#if (UNIX)
  #add_library(${LIBRARY_TARGET_NAME} SHARED ${${LIBRARY_TARGET_NAME}_SRC})
#endif(UNIX)
#target_link_libraries(${LIBRARY_TARGET_NAME} PRIVATE Torch)
##target_compile_options(${LIBRARY_TARGET_NAME} PUBLIC)

add_executable(example-app main.cpp NMS.cpp)
target_link_libraries(example-app  ${TORCH_LIBRARIES})
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)
