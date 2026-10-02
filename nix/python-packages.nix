# Imported by flake.nix, shell.nix, and default.nix to define the Python packages.
{ pkgs, pythonPackages }:

let
  inherit (pkgs) lib stdenv;

  imguiBundleVersion = "1.92.5";
  imguiBundleWheels = {
    "3.11" = {
      x86_64-linux = {
        filename = "imgui_bundle-${imguiBundleVersion}-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        url = "https://files.pythonhosted.org/packages/44/5c/a698a22058c575dc0097429e20f5c26c023c89ece922168e459ecd947c1a/imgui_bundle-${imguiBundleVersion}-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-ubq7fQUbttMY6FlYPZSZrnaxzOhF+wBTjM0UP0xaH/k=";
      };
      aarch64-darwin = {
        filename = "imgui_bundle-${imguiBundleVersion}-cp311-cp311-macosx_14_0_arm64.whl";
        url = "https://files.pythonhosted.org/packages/79/ae/292c0b89ae1967bfdaea3e43cadc62ca34a3e3fbec4f307a87e01ad9ef56/imgui_bundle-${imguiBundleVersion}-cp311-cp311-macosx_14_0_arm64.whl";
        hash = "sha256-u79j+DJFxRquyTpbAQp3DxSMsJ30jfr1IkyoRwU32vQ=";
      };
      x86_64-darwin = {
        filename = "imgui_bundle-${imguiBundleVersion}-cp311-cp311-macosx_14_0_x86_64.whl";
        url = "https://files.pythonhosted.org/packages/d5/c5/9c01a50760ae9d5e62d760659227356601a76f75292e7de496d9ad8f74e3/imgui_bundle-${imguiBundleVersion}-cp311-cp311-macosx_14_0_x86_64.whl";
        hash = "sha256-abm3bbHRIIhfzc1ocHzAq+L27p3Ni5/jvKK2yqtR8Cc=";
      };
    };
    "3.12" = {
      x86_64-linux = {
        filename = "imgui_bundle-${imguiBundleVersion}-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        url = "https://files.pythonhosted.org/packages/19/2f/7b0bd3074fb906d668bc733641d7ff24729ebeeab7bbfbaeb4d6b5052890/imgui_bundle-${imguiBundleVersion}-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-fX3H8MCQ/CfP3FsFWiNYZPjZSNjeZNukL9/RALAX1yg=";
      };
      aarch64-darwin = {
        filename = "imgui_bundle-${imguiBundleVersion}-cp312-cp312-macosx_14_0_arm64.whl";
        url = "https://files.pythonhosted.org/packages/8e/7e/2ac14f99a8464ffadc753b21047ef9932d2c74b4a3706dcaecaeff4fdc3e/imgui_bundle-${imguiBundleVersion}-cp312-cp312-macosx_14_0_arm64.whl";
        hash = "sha256-at2OP9h+Rudjw+9HH/UfnuMfWcgsTZeEeUXzOYYNIfE=";
      };
      x86_64-darwin = {
        filename = "imgui_bundle-${imguiBundleVersion}-cp312-cp312-macosx_14_0_x86_64.whl";
        url = "https://files.pythonhosted.org/packages/2b/12/1b05988231ccf5bcf1d391a9d0a29acd18b871ad24a7fe2c3ca9ca54c83a/imgui_bundle-${imguiBundleVersion}-cp312-cp312-macosx_14_0_x86_64.whl";
        hash = "sha256-tfjSmnfNBhGqTM1EWFj7JV8LizTbzg6oWpTf1n9DRgc=";
      };
    };
    "3.13" = {
      x86_64-linux = {
        filename = "imgui_bundle-${imguiBundleVersion}-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        url = "https://files.pythonhosted.org/packages/e1/a7/ff3c10d708f543092fab73f553a86e9cd12cb70dd4461f28ecacf8becb97/imgui_bundle-${imguiBundleVersion}-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-6eTRvwkClCeas+NUGjHLR9qY0i+1DLnwWwfUqNur3Ug=";
      };
      aarch64-darwin = {
        filename = "imgui_bundle-${imguiBundleVersion}-cp313-cp313-macosx_14_0_arm64.whl";
        url = "https://files.pythonhosted.org/packages/1c/9d/f490ac25d03bf3d914535ce95cfa5f390403fbc2519dd777e27f8ec0459a/imgui_bundle-${imguiBundleVersion}-cp313-cp313-macosx_14_0_arm64.whl";
        hash = "sha256-FlKebsqn9/Ylk2HGc4p+k17qoJIfuWRts52Kz3l4Ups=";
      };
      x86_64-darwin = {
        filename = "imgui_bundle-${imguiBundleVersion}-cp313-cp313-macosx_14_0_x86_64.whl";
        url = "https://files.pythonhosted.org/packages/0a/1c/c3052f2d9c3958f6b22c248af349a01c5f4917e290734bbbffd37cc122ca/imgui_bundle-${imguiBundleVersion}-cp313-cp313-macosx_14_0_x86_64.whl";
        hash = "sha256-82nSp2+HfV8eC9ZbaUL/+UYfyquAJ6CAqBHv4yEh+Zk=";
      };
    };
    "3.14" = {
      x86_64-linux = {
        filename = "imgui_bundle-${imguiBundleVersion}-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        url = "https://files.pythonhosted.org/packages/64/26/9fc81b06590b145fe5f56c020febb01bf366851038f2992673b794ebf196/imgui_bundle-${imguiBundleVersion}-cp314-cp314-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-gmHKgaawtOKH/iEdMPZj79S9O6+4psxti9MyNOAItrk=";
      };
      aarch64-darwin = {
        filename = "imgui_bundle-${imguiBundleVersion}-cp314-cp314-macosx_14_0_arm64.whl";
        url = "https://files.pythonhosted.org/packages/00/1c/fb495669914483d81af938bf1968cf3b642a0432fa8086ec9d9ef2563e62/imgui_bundle-${imguiBundleVersion}-cp314-cp314-macosx_14_0_arm64.whl";
        hash = "sha256-XCxvKt3hUrLTV2eIaoOk9asfUHbYmIfGPqd6rdTUx4c=";
      };
      x86_64-darwin = {
        filename = "imgui_bundle-${imguiBundleVersion}-cp314-cp314-macosx_14_0_x86_64.whl";
        url = "https://files.pythonhosted.org/packages/23/a7/94063bfdd8a2d227aedb1893bb4bac3d1a8cfe0f50a93ce9040929015aa0/imgui_bundle-${imguiBundleVersion}-cp314-cp314-macosx_14_0_x86_64.whl";
        hash = "sha256-X/xkGtglfm67Iv89UHASz8u51/+40Cr74sFkl0bSMwM=";
      };
    };
  };

  pythonVersion = lib.versions.majorMinor pythonPackages.python.version;
  imguiBundleWheelsForPython = imguiBundleWheels.${pythonVersion} or
    (throw "pyviewer: imgui-bundle ${imguiBundleVersion} has no pinned wheel for Python ${pythonVersion}");
  imguiBundleWheel = imguiBundleWheelsForPython.${stdenv.hostPlatform.system} or
    (throw "pyviewer: imgui-bundle ${imguiBundleVersion} has no pinned Python ${pythonVersion} wheel for ${stdenv.hostPlatform.system}");

  imguiBundle = pythonPackages.buildPythonPackage {
    pname = "imgui-bundle";
    version = imguiBundleVersion;
    format = "wheel";

    src = pkgs.fetchurl {
      name = imguiBundleWheel.filename;
      inherit (imguiBundleWheel) url hash;
    };

    nativeBuildInputs = lib.optionals stdenv.hostPlatform.isLinux [
      pkgs.autoPatchelfHook
    ];

    buildInputs = lib.optionals stdenv.hostPlatform.isLinux [
      stdenv.cc.cc.lib
      pkgs.glfw
      pkgs.libx11
      pkgs.libxext
      pkgs.zlib
    ];

    # Use the same GLFW as pythonPackages.glfw instead of the wheel's bundled 3.3.
    postInstall = lib.optionalString stdenv.hostPlatform.isLinux ''
      bundleDir="$out/${pythonPackages.python.sitePackages}/imgui_bundle"
      rm "$bundleDir"/libglfw.so*
      ln -s ${pkgs.glfw}/lib/libglfw.so.3 "$bundleDir/libglfw.so.3"
    '';

    dependencies = [ pythonPackages.numpy ];
    pythonImportsCheck = [ "imgui_bundle" ];
  };

  lightProcess = pythonPackages.buildPythonPackage rec {
    pname = "light-process";
    version = "0.0.7";
    format = "setuptools";

    src = pkgs.fetchPypi {
      pname = "light_process";
      inherit version;
      hash = "sha256-td9iap4uukBs1i48T/XS6dJjVIxmR3eW92uPQO1orLw=";
    };

    nativeBuildInputs = [ pythonPackages.setuptools ];
    pythonImportsCheck = [ "light_process" ];
  };

  pythonWayland = pythonPackages.buildPythonPackage rec {
    pname = "python-wayland";
    version = "1.0.0";
    pyproject = true;

    src = pkgs.fetchPypi {
      pname = "python_wayland";
      inherit version;
      hash = "sha256-Jfr58ku9FCiUdi4TbC1yzUeOdGZRrQ7r6aamr1eU6+A=";
    };

    build-system = [ pythonPackages.hatchling ];
    pythonImportsCheck = [ "wayland" ];
  };

  source = builtins.path {
    path = ../.;
    name = "pyviewer-source";
    filter = path: type:
      let name = baseNameOf path;
      in !(builtins.elem name [
        ".direnv"
        ".git"
        ".nix-python"
        ".venv"
        ".vscode"
        "build"
        "dist"
        "result"
      ])
      && !(lib.hasSuffix ".egg-info" name)
      && !(lib.hasSuffix ".ini" name);
  };

  pyviewer = pythonPackages.buildPythonPackage {
    pname = "pyviewer";
    version = "2.1.0";
    pyproject = true;
    src = source;

    build-system = [ pythonPackages.setuptools ];
    dependencies = [
      pythonPackages.glfw
      pythonPackages.numpy
      pythonPackages.pyopengl
      imguiBundle
      pythonPackages.setuptools
      lightProcess
      pythonPackages.py
    ] ++ lib.optionals stdenv.hostPlatform.isLinux [
      pythonWayland
    ];

    pythonImportsCheck = [
      "pyviewer"
      "pyviewer.utils"
    ];

    meta = {
      description = "Interactive Python viewers";
      homepage = "https://github.com/harskish/pyviewer";
      license = lib.licenses.cc-by-nc-sa-40;
      platforms = builtins.attrNames imguiBundleWheelsForPython;
    };
  };
in
{
  imgui-bundle = imguiBundle;
  light-process = lightProcess;
  python-wayland = pythonWayland;
  inherit pyviewer;
}
