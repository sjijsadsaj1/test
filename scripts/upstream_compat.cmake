# Restore Boost's legacy global bind placeholders for unmodified upstream C++.
# This changes name lookup only, not planner parameters or algorithm behavior.
add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:-include>" "$<$<COMPILE_LANGUAGE:CXX>:boost/bind.hpp>")

# Debian's log4cxx headers require C++17. Apply after the upstream project has
# declared its targets, so its older CMAKE_CXX_STANDARD setting cannot undo it.
function(upstream_require_cxx17)
  get_property(local_targets DIRECTORY PROPERTY BUILDSYSTEM_TARGETS)
  foreach(local_target IN LISTS local_targets)
    get_target_property(local_type ${local_target} TYPE)
    if(local_type MATCHES "^(EXECUTABLE|SHARED_LIBRARY|STATIC_LIBRARY|OBJECT_LIBRARY)$")
      set_property(TARGET ${local_target} PROPERTY CXX_STANDARD 17)
      set_property(TARGET ${local_target} PROPERTY CXX_STANDARD_REQUIRED ON)
    endif()
  endforeach()
endfunction()
cmake_language(DEFER CALL upstream_require_cxx17)
