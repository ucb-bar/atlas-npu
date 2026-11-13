########################################################################################################################
# Atlas baremetal tests
#
# One executable per assembly/*.S when tests/atlas_<name>.c exists (from assembler.py).
#   e.g. assembly/alu_basic.S  →  atlas_alu_basic.c  →  target atlas_alu_basic
#
# Build instructions:
#   cmake -S ./ -B ./build/ -D CMAKE_BUILD_TYPE=Debug
#   cmake --build ./build/ --target atlas_alu_basic
# Cleaning:
#   cmake --build ./build/ --target clean
########################################################################################################################
cmake_minimum_required(VERSION 3.10)

project(atlas-baremetal LANGUAGES C)


#################################
# RISCV Toolchain
#################################

set(CMAKE_SYSTEM_NAME         "Generic" CACHE STRING "")
set(CMAKE_SYSTEM_PROCESSOR    "riscv"   CACHE STRING "")

set(TOOLCHAIN_PREFIX    "riscv64-unknown-elf-")

set(CMAKE_AR            "${TOOLCHAIN_PREFIX}ar")
set(CMAKE_ASM_COMPILER  "${TOOLCHAIN_PREFIX}gcc")
set(CMAKE_C_COMPILER    "${TOOLCHAIN_PREFIX}gcc")
set(CMAKE_CXX_COMPILER  "${TOOLCHAIN_PREFIX}g++")
set(CMAKE_LINKER        "${TOOLCHAIN_PREFIX}ld")
set(CMAKE_OBJCOPY       "${TOOLCHAIN_PREFIX}objcopy")
set(CMAKE_OBJDUMP       "${TOOLCHAIN_PREFIX}objdump")
set(CMAKE_SIZE          "${TOOLCHAIN_PREFIX}size")

set(CMAKE_EXECUTABLE_SUFFIX            ".riscv")


#################################
# Flags
#################################

set(ARCH                "rv64imafd")
set(ABI                 "lp64d")
set(CMODEL              "medany")
set(ARCH_FLAGS          -march=${ARCH} -mabi=${ABI} -mcmodel=${CMODEL})

set(SPECS               "htif_nano.specs")
set(SPEC_FLAGS          -specs=${SPECS})

set(LINKER_SCRIPT       "htif.ld")

add_compile_options(-std=gnu99)
add_compile_options(-O2 -Wall -Wextra)
add_compile_options(-fno-common -fno-builtin-printf)
add_compile_options(${ARCH_FLAGS})
add_compile_options(${SPEC_FLAGS})

add_link_options(-static)
add_link_options(${SPEC_FLAGS})
add_link_options(-T ${LINKER_SCRIPT})


#################################
# Disassembly helper
#################################

add_custom_target(dump ALL)

function(add_dump_target target_name)
    add_custom_target(${target_name}-dump
        BYPRODUCTS ${CMAKE_BINARY_DIR}/${target_name}.dump
        COMMAND ${CMAKE_OBJDUMP} -D $<TARGET_FILE:${target_name}> > ${CMAKE_BINARY_DIR}/${target_name}.dump
        DEPENDS ${target_name}
        COMMENT "Generating disassembly for ${target_name}"
    )
    add_dependencies(${target_name}-dump ${target_name})
    add_dependencies(dump ${target_name}-dump)
endfunction()


#################################
# Executables — atlas_<stem> from ../assembly/<stem>.S + atlas_<stem>.c
#################################


set(PMU_REPORT_GROUPS "all" CACHE STRING "Comma-separated PMU event prefixes, or all")
if(NOT PMU_REPORT_GROUPS MATCHES "^[A-Za-z0-9_]+(\\.[A-Za-z0-9_]+)*(,[A-Za-z0-9_]+(\\.[A-Za-z0-9_]+)*)*$")
  message(FATAL_ERROR "PMU_REPORT_GROUPS must be comma-separated dot-delimited names, or all")
endif()

option(PMU_ENABLE "Enable Atlas tile PMU collection and final counter dump" OFF)
option(PMU_POLL_ENABLE "Snapshot the Atlas PMU during execution" OFF)
set(PMU_POLL_COUNTER_ID 0 CACHE STRING "PMU counter ID or generated event macro to poll")
set(ATLAS_PMU_HEADER "" CACHE FILEPATH "Optional hardware-generated PMU event/address header")
set(PMU_POLL_INTERVAL 1000 CACHE STRING "Execution polls between PMU snapshots")
option(PMU_POLL_PRINT "Print each PMU polling sample" OFF)
if(PMU_POLL_ENABLE AND NOT PMU_ENABLE)
  message(FATAL_ERROR "PMU_POLL_ENABLE requires PMU_ENABLE")
endif()
if(PMU_POLL_ENABLE AND NOT PMU_POLL_INTERVAL MATCHES "^[1-9][0-9]*$")
  message(FATAL_ERROR "PMU_POLL_INTERVAL must be a positive integer")
endif()


# Generate once per hardware configuration, never during target execution.
set(PMU_MANIFEST "" CACHE FILEPATH "Schema-v2 tile PMU event manifest")
set(PMU_PLATFORM "" CACHE FILEPATH "Matching Chipyard JSON device tree")
set(PMU_SOFTWARE_DIR "${CMAKE_CURRENT_LIST_DIR}/../../../pmu/software" CACHE PATH "pmu/software directory")
if(PMU_MANIFEST OR PMU_PLATFORM)
  if(NOT PMU_MANIFEST OR NOT PMU_PLATFORM)
    message(FATAL_ERROR "Set both PMU_MANIFEST and PMU_PLATFORM")
  endif()
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(_pmu_header "${CMAKE_CURRENT_BINARY_DIR}/tile_pmu_platform.h")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${PMU_MANIFEST}" "${PMU_PLATFORM}" "${PMU_SOFTWARE_DIR}/generate_platform_header.py")
  execute_process(COMMAND "${Python3_EXECUTABLE}" "${PMU_SOFTWARE_DIR}/generate_platform_header.py"
    --manifest "${PMU_MANIFEST}" --platform "${PMU_PLATFORM}" --output "${_pmu_header}"
    RESULT_VARIABLE _pmu_result)
  if(NOT _pmu_result EQUAL 0)
    message(FATAL_ERROR "PMU platform header generation failed")
  endif()
  set(ATLAS_PMU_HEADER "${_pmu_header}")
endif()

if(PMU_ENABLE AND NOT PMU_REPORT_GROUPS STREQUAL "all" AND NOT ATLAS_PMU_HEADER)
  message(FATAL_ERROR "PMU_REPORT_GROUPS requires ATLAS_PMU_HEADER or PMU_MANIFEST and PMU_PLATFORM")
endif()

file(GLOB_RECURSE _ATLAS_ASM "${CMAKE_CURRENT_SOURCE_DIR}/../assembly/*.S")
list(SORT _ATLAS_ASM)

set(_atlas_n 0)
set(_atlas_skip "")
set(_atlas_combo_targets "")
set(_atlas_smoke_targets "")
set(_atlas_pairwise_targets "")
set(_atlas_risk_targets "")
set(_atlas_stress_targets "")
set(_atlas_perf_targets "")

foreach(_asm IN LISTS _ATLAS_ASM)
    get_filename_component(_stem ${_asm} NAME_WE)
    set(_c "${CMAKE_CURRENT_SOURCE_DIR}/atlas_${_stem}.c")
    set(_tgt "atlas_${_stem}")

    if(EXISTS ${_c})
        add_executable(${_tgt} ${_c})
  target_compile_definitions(${_tgt} PRIVATE
    PMU_REPORT_GROUPS="${PMU_REPORT_GROUPS}"
    PMU_ENABLE=$<BOOL:${PMU_ENABLE}> PMU_POLL_ENABLE=$<BOOL:${PMU_POLL_ENABLE}>
    PMU_POLL_COUNTER_ID=${PMU_POLL_COUNTER_ID} PMU_POLL_INTERVAL=${PMU_POLL_INTERVAL}
    PMU_POLL_PRINT=$<BOOL:${PMU_POLL_PRINT}>)
  if(ATLAS_PMU_HEADER)
    target_include_directories(${_tgt} PRIVATE "${PMU_SOFTWARE_DIR}")
    target_compile_definitions(${_tgt} PRIVATE ATLAS_PMU_HEADER="${ATLAS_PMU_HEADER}")
  endif()

        add_dump_target(${_tgt})
        math(EXPR _atlas_n "${_atlas_n}+1")
        list(APPEND _atlas_combo_targets ${_tgt})

        if(_stem MATCHES "^smk_")
            list(APPEND _atlas_smoke_targets ${_tgt})
        elseif(_stem MATCHES "^pw_")
            list(APPEND _atlas_pairwise_targets ${_tgt})
        elseif(_stem MATCHES "^rsk_")
            list(APPEND _atlas_risk_targets ${_tgt})
        elseif(_stem MATCHES "^str_")
            list(APPEND _atlas_stress_targets ${_tgt})
        elseif(_stem MATCHES "^perf_")
            list(APPEND _atlas_perf_targets ${_tgt})
        endif()
    else()
        list(APPEND _atlas_skip "${_stem}")
    endif()
endforeach()

if(_atlas_skip)
    string(REPLACE ";" ", " _skip_msg "${_atlas_skip}")
    message(STATUS "Atlas baremetal: ${_atlas_n} executable(s); no atlas_*.c yet (skipped): ${_skip_msg}")
else()
    message(STATUS "Atlas baremetal: ${_atlas_n} executable(s) from assembly/*.S")
endif()

add_custom_target(atlas-combo)
foreach(_t IN LISTS _atlas_combo_targets)
    add_dependencies(atlas-combo ${_t})
endforeach()

add_custom_target(atlas-smoke)
foreach(_t IN LISTS _atlas_smoke_targets)
    add_dependencies(atlas-smoke ${_t})
endforeach()

add_custom_target(atlas-pairwise)
foreach(_t IN LISTS _atlas_pairwise_targets)
    add_dependencies(atlas-pairwise ${_t})
endforeach()

add_custom_target(atlas-risk)
foreach(_t IN LISTS _atlas_risk_targets)
    add_dependencies(atlas-risk ${_t})
endforeach()

add_custom_target(atlas-stress)
foreach(_t IN LISTS _atlas_stress_targets)
    add_dependencies(atlas-stress ${_t})
endforeach()

add_custom_target(atlas-perf)
foreach(_t IN LISTS _atlas_perf_targets)
    add_dependencies(atlas-perf ${_t})
endforeach()
