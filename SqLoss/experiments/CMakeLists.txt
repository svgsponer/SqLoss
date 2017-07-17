include_directories(${SEQL_SOURCE_DIR})

add_executable(sqloss_regression SeqlRegression.cpp)
add_executable(sqloss_mmc MMC.cpp)
add_executable(sqloss_dream Dream5.cpp)

target_link_libraries(sqloss_regression Sq_learn SNode LinearModel Sq_predict common_string)
target_link_libraries(sqloss_mmc Sq_learn SNode LinearModel Sq_predict common_string)
target_link_libraries(sqloss_dream Sq_learn SNode LinearModel Sq_predict common_string)
