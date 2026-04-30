#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "SofaPython3Testing" for configuration "Release"
set_property(TARGET SofaPython3Testing APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(SofaPython3Testing PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libSofaPython3Testing.so.1.0"
  IMPORTED_SONAME_RELEASE "libSofaPython3Testing.so.1.0"
  )

list(APPEND _cmake_import_check_targets SofaPython3Testing )
list(APPEND _cmake_import_check_files_for_SofaPython3Testing "${_IMPORT_PREFIX}/lib/libSofaPython3Testing.so.1.0" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
