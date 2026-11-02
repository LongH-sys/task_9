aux_source_directory(src LIB_SRC_FILES)
add_library(my_lib SHARED ${LIB_SRC_FILES})
target_include_directories(my_lib PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_link_libraries(my_lib PRIVATE camera ${OpenCV_LIBS} Eigen3::Eigen
                            PUBLIC YoloVino_LIB)