#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaPython3::Bindings.SofaTypes" for configuration "Release"
set_property(TARGET SofaPython3::Bindings.SofaTypes APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaPython3::Bindings.SofaTypes PROPERTIES
  IMPORTED_LINK_DEPENDENT_LIBRARIES_RELEASE "Python::Python"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/SofaTypes.cpython-310-x86_64-linux-gnu.so"
  IMPORTED_SONAME_RELEASE "SofaTypes.cpython-310-x86_64-linux-gnu.so"
  )

list(APPEND _cmake_import_check_targets SofaPython3::Bindings.SofaTypes )
list(APPEND _cmake_import_check_files_for_SofaPython3::Bindings.SofaTypes "${_IMPORT_PREFIX}/lib/SofaTypes.cpython-310-x86_64-linux-gnu.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
