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
