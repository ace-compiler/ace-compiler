find_package (SEAL QUIET)
if (SEAL_FOUND)
	message (STATUS "Build FHErtlib with SEAL support")
	get_target_property(SEAL_INCLUDE_DIR SEAL::seal INTERFACE_INCLUDE_DIRECTORIES)
else ()
	message (WARNING "SEAL library not found, no SEAL support in FHErtlib")
	return ()
endif ()

file (GLOB_RECURSE RT_SEAL_SRC_FILES CONFIGURE_DEPENDS src/*.cxx)

set (RT_SEAL_INCLUDE_DIRS "")
list (APPEND RT_SEAL_INCLUDE_DIRS "${PROJECT_SOURCE_DIR}/include")
list (APPEND RT_SEAL_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include")
list (APPEND RT_SEAL_INCLUDE_DIRS "${SEAL_INCLUDE_DIR}")
include_directories (${RT_SEAL_INCLUDE_DIRS})
set (RT_SEAL_INCLUDE_DIRS "${RT_SEAL_INCLUDE_DIRS}" CACHE INTERNAL "")

add_library (fhert_seal_obj OBJECT ${RT_SEAL_SRC_FILES})
set_property (TARGET fhert_seal_obj PROPERTY POSITION_INDEPENDENT_CODE 1)

if (BUILD_STATIC)
	add_library (FHErt_seal STATIC $<TARGET_OBJECTS:fhert_seal_obj>)
	set_property (TARGET FHErt_seal PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/rtlib/lib)
	install (TARGETS FHErt_seal EXPORT FHETargets DESTINATION rtlib/lib)
endif ()

add_custom_target (fhert_seal_all)

if (BUILD_STATIC)
	set (FHERT_SEAL_LIB ${FHERT_SEAL_LIB} PUBLIC FHErt_seal SEAL::seal FHErt_common)
	add_dependencies (fhert_seal_all FHErt_seal)
endif()

set (FHERT_SEAL_UTAPP "")
if (BUILD_UNITTEST)
	file (GLOB FHERT_SEAL_UNITTEST_SRC_FILES CONFIGURE_DEPENDS unittest/*.cxx)
	set (FHERT_SEAL_UNITTEST_SRC_FILES ${FHERT_SEAL_UNITTEST_SRC_FILES})
	add_executable (ut_fhert_seal ${FHERT_SEAL_UNITTEST_SRC_FILES} ${UNITTESTMAIN})
	set_property (TARGET ut_fhert_seal PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unittest)
	target_link_libraries (ut_fhert_seal ${FHERT_SEAL_LIB} ${FHE_UTLIBS} ${MATH_LIBS})
	set (FHERT_SEAL_UTAPP ${FHERT_SEAL_UTAPP} ut_fhert_seal)

	add_dependencies (fhert_seal_all ut_fhert_seal)

	add_custom_command (OUTPUT run_fhert_seal_utapp WORKING_DIRECTORY ${CMAKE_BINARY_DIR} COMMAND ${CMAKE_BINARY_DIR}/unittest/ut_fhert_seal)
	add_custom_target (test_fhert_seal_ut DEPENDS ut_fhertlib run_fhert_seal_utapp)
	add_test( NAME ut_fhert_seal COMMAND ${CMAKE_BINARY_DIR}/unittest/ut_fhert_seal)
endif()

set (FHERT_SEAL_EGAPP "")
if (FHE_BUILD_EXAMPLE)
	file (GLOB FHERT_SEAL_EXAMPLE_SRC_FILES CONFIGURE_DEPENDS example/*.cxx)
	foreach (app ${FHERT_SEAL_EXAMPLE_SRC_FILES})
		get_filename_component (exe ${app} NAME_WE)
		add_executable (${exe} ${app})
		set_property (TARGET ${exe} PROPERTY RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/example)
		set (FHERT_SEAL_EGAPP ${FHERT_SEAL_EGAPP} ${exe})
		target_link_libraries (${exe} ${FHERT_SEAL_LIB} ${NN_LIBS} ${MATH_LIBS})
		# add_test( NAME ${exe} COMMAND ${CMAKE_BINARY_DIR}/example/${exe})
	endforeach ()

	add_custom_target (fhert_seal_example)
	add_dependencies (fhert_seal_example ${FHERT_SEAL_EGAPP})
	add_dependencies (fhert_seal_all fhert_seal_example)
endif ()

if (FHE_INSTALL_APP)
	install (TARGETS ${FHERT_SEAL_UTAPP} RUNTIME DESTINATION unittest)
	install (TARGETS ${FHERT_SEAL_EGAPP} RUNTIME DESTINATION example)
endif ()
