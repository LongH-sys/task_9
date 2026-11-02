add_executable(main src/main.cpp)
target_link_libraries(main PRIVATE 
    my_lib
    camera 
    ${OpenCV_LIBS}
    Eigen3::Eigen)