workspace "COMP5822M-exercises"
	language "C++"
	cppdialect "C++17"

	platforms { "x64" }
	configurations { "debug", "release" }

	flags "NoPCH"
	flags "MultiProcessorCompile"

	startproject "cw4"

	debugdir "%{wks.location}"
	objdir "_build_/%{cfg.buildcfg}-%{cfg.platform}-%{cfg.toolset}"
	targetsuffix "-%{cfg.buildcfg}-%{cfg.platform}-%{cfg.toolset}"
	
	-- Default toolset options
	filter "toolset:gcc or toolset:clang"
		linkoptions { "-pthread" }
		buildoptions { "-march=native", "-Wall", "-pthread" }

	filter "toolset:msc-*"
		defines { "_CRT_SECURE_NO_WARNINGS=1" }
		defines { "_SCL_SECURE_NO_WARNINGS=1" }
		buildoptions { "/utf-8" }
	
	filter "*"

	-- default libraries
	filter "system:linux"
		links "dl"
	
	filter "system:windows"

	filter "*"

	-- default outputs
	filter "kind:StaticLib"
		targetdir "lib/"

	filter "kind:ConsoleApp"
		targetdir "bin/"
		targetextension ".exe"
	
	filter "*"

	--configurations
	filter "debug"
		symbols "On"
		defines { "_DEBUG=1" }

	filter "release"
		optimize "On"
		defines { "NDEBUG=1" }

	filter "*"

-- Third party dependencies
include "third_party" 

-- GLSLC helpers
dofile( "util/glslc.lua" )

-- Projects
project "cw4"
	local sources = { 
		"cw4/**.cpp",
		"cw4/**.hpp",
		"cw4/**.hxx"
	}

	kind "ConsoleApp"
	location "cw4"

	files( sources )

	dependson "cw4-shaders"

	links "labutils"
	links "x-volk"
	links "x-stb"
	links "x-glfw"
	links "x-vma"

	dependson "x-glm" 

project "cw4-shaders"
	local shaders = { 
		"cw4/shaders/*.vert",
		"cw4/shaders/*.frag",
		"cw4/shaders/*.comp",
		"cw4/shaders/*.geom",
		"cw4/shaders/*.tesc",
		"cw4/shaders/*.tese"
	}

	kind "Utility"
	location "cw4/shaders"

	files( shaders )

	handle_glsl_files( "-O", "assets/cw4/shaders", {} )

project "cw4-bake"
	local sources = { 
		"cw4-bake/**.cpp",
		"cw4-bake/**.hpp",
		"cw4-bake/**.hxx"
	}

	kind "ConsoleApp"
	location "cw4-bake"

	files( sources )

	links "labutils" -- for lut::Error
	links "x-tgen"
	links "x-zstd"

	dependson "x-glm" 
	dependson "x-rapidobj"


project "labutils"
	local sources = { 
		"labutils/**.cpp",
		"labutils/**.hpp",
		"labutils/**.hxx"
	}

	kind "StaticLib"
	location "labutils"

	files( sources )

--EOF
