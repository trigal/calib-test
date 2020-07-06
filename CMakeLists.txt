project(calib-test)
cmake_minimum_required(VERSION 2.8)
set (CMAKE_CXX_STANDARD 11)
set (CMAKE_INCLUDE_DIRECTORIES_BEFORE on)
include_directories(BEFORE "/usr/local/include/opencv4")

set(OpenCV_DIR "/usr/local/lib")
find_package(OpenCV 4.3 REQUIRED COMPONENTS core imgproc calib3d highgui EXACT REQUIRED PATHS /usr/local/lib)

message(GIGI ${OpenCV_LIBS} )

# aux_source_directory(. SRC_LIST)
# add_executable(${PROJECT_NAME} ${SRC_LIST})
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

add_executable(calib-test main.cpp)
target_link_libraries(calib-test ${OpenCV_LIBS})

add_executable(projectpoints projectpoints.cpp)
target_link_libraries(projectpoints ${OpenCV_LIBS})
