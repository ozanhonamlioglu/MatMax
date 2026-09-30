add_executable(logreg
  ml/logistic-regression.cpp
)

target_link_libraries(logreg PRIVATE matmax)

add_executable(logreg_fast
  ml/logistic-regression-fast.cpp
)

target_link_libraries(logreg_fast PRIVATE matmax)
