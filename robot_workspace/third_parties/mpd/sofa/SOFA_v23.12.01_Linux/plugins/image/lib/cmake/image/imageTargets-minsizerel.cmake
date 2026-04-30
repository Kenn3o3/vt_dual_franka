#----------------------------------------------------------------
# Generated CMake target import file for configuration "MinSizeRel".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "image_gui" for configuration "MinSizeRel"
set_property(TARGET image_gui APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(image_gui PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libimage_gui.so.23.12.01"
  IMPORTED_SONAME_MINSIZEREL "libimage_gui.so.23.12.01"
  )

list(APPEND _IMPORT_CHECK_TARGETS image_gui )
list(APPEND _IMPORT_CHECK_FILES_FOR_image_gui "${_IMPORT_PREFIX}/lib/libimage_gui.so.23.12.01" )

# Import target "image" for configuration "MinSizeRel"
set_property(TARGET image APPEND PROPERTY IMPORTED_CONFIGURATIONS MINSIZEREL)
set_target_properties(image PROPERTIES
  IMPORTED_LOCATION_MINSIZEREL "${_IMPORT_PREFIX}/lib/libimage.so.0.1"
  IMPORTED_SONAME_MINSIZEREL "libimage.so.0.1"
  )

list(APPEND _IMPORT_CHECK_TARGETS image )
list(APPEND _IMPORT_CHECK_FILES_FOR_image "${_IMPORT_PREFIX}/lib/libimage.so.0.1" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
