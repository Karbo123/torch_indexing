cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(libtorch_indexing LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
find_package(Torch REQUIRED)
find_package(PythonLibs REQUIRED)
find_package(TBB REQUIRED)
find_package(Thrust REQUIRED) # finding thrust
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fno-gnu-unique") # https://github.com/pytorch/pytorch/issues/52663#issuecomment-809092114

thrust_create_target(Thrust HOST TBB DEVICE CUDA)

file(GLOB SOURCES "/data/lei.jiabao/NIPS2021/recon/lib/torch_indexing/csrc/batch_sort.cu")
add_library(${PROJECT_NAME} SHARED ${SOURCES})

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)
set_target_properties(${PROJECT_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(${PROJECT_NAME} PROPERTIES PREFIX "")

target_link_libraries(${PROJECT_NAME} ${TORCH_PYTHON_LIBRARY})
target_link_libraries(${PROJECT_NAME} Thrust) # link to thrust

target_compile_definitions(${PROJECT_NAME} PUBLIC TORCH_EXTENSION_NAME=${PROJECT_NAME})
target_compile_definitions(${PROJECT_NAME} PUBLIC TORCH_API_INCLUDE_EXTENSION_H)
target_compile_definitions(${PROJECT_NAME} PUBLIC ${TORCH_CXX_FLAGS})

target_include_directories(${PROJECT_NAME} PUBLIC ${TORCH_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${PYTHON_INCLUDE_DIRS})
target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})


# to install
string(REGEX REPLACE "lib" "" BARE_PROJECT_NAME ${PROJECT_NAME})
install(CODE "execute_process(                                        \
    COMMAND ${CMAKE_COMMAND} -E create_symlink                        \
    ${PROJECT_SOURCE_DIR}/${BARE_PROJECT_NAME}                        \
    ${CMAKE_INSTALL_PREFIX}/${BARE_PROJECT_NAME}                      \
    )"
)
install(CODE "execute_process(                                        \
    COMMAND ${CMAKE_COMMAND} -E create_symlink                        \
    ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.so                            \
    ${PROJECT_SOURCE_DIR}/${BARE_PROJECT_NAME}/src/${PROJECT_NAME}.so \
    )"
)
