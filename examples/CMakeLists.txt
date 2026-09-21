add_executable(logreg
  ml/logistic-regression.cpp
)

target_link_libraries(logreg PRIVATE matmax)
