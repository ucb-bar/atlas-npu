# Shared by Atlas baremetal and Chipyard's Saturn tests. Each build tree owns
# its generated headers, so parallel builds never rewrite source-tree images.
set(ATLAS_DISPATCHER_DIR "${CMAKE_CURRENT_LIST_DIR}")
set(ATLAS_DISPATCHER_TEST_DIR "${ATLAS_DISPATCHER_DIR}/../tests/dispatcher")
set(ATLAS_DISPATCHER_IMAGES "${CMAKE_CURRENT_BINARY_DIR}/dispatcher_images")
find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_custom_command(
    OUTPUT "${ATLAS_DISPATCHER_IMAGES}/dispatcher_image.h"
           "${ATLAS_DISPATCHER_IMAGES}/test_kernel_image.h"
    COMMAND "${Python3_EXECUTABLE}" "${ATLAS_DISPATCHER_DIR}/generate.py"
            --out-dir "${ATLAS_DISPATCHER_IMAGES}"
            --test-kernel "${ATLAS_DISPATCHER_TEST_DIR}/test_kernel.S"
    DEPENDS "${ATLAS_DISPATCHER_DIR}/generate.py"
            "${ATLAS_DISPATCHER_DIR}/dispatcher.S"
            "${ATLAS_DISPATCHER_TEST_DIR}/test_kernel.S"
            "${ATLAS_DISPATCHER_DIR}/../assembler.py"
    COMMENT "Assembling resident Atlas dispatcher and test kernel"
    VERBATIM)
add_custom_target(atlas_dispatcher_images DEPENDS
    "${ATLAS_DISPATCHER_IMAGES}/dispatcher_image.h"
    "${ATLAS_DISPATCHER_IMAGES}/test_kernel_image.h")

function(atlas_use_dispatcher target)
    add_dependencies(${target} atlas_dispatcher_images)
    target_include_directories(${target} BEFORE PRIVATE
        "${ATLAS_DISPATCHER_IMAGES}" "${ATLAS_DISPATCHER_DIR}")
endfunction()
