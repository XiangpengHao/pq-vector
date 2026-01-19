{
  description = "Liquid Cache Flake Configuration";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    rust-overlay.url = "github:oxalica/rust-overlay";
    flake-utils.url = "github:numtide/flake-utils";
    crane.url = "github:ipetkov/crane";
  };

  outputs =
    {
      nixpkgs,
      rust-overlay,
      flake-utils,
      crane,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        overlays = [ (import rust-overlay) ];
        pkgs = import nixpkgs {
          inherit system overlays;
        };
        craneLib = crane.mkLib pkgs;
      in
      {
        devShells.default =
          with pkgs;
          mkShell {
            packages = [
              openssl
              pkg-config
              eza
              fd
              llvmPackages.bintools
              lldb
              nixd
              (rust-bin.selectLatestNightlyWith (
                toolchain:
                toolchain.default.override {
                  extensions = [
                    "rust-src"
                    "llvm-tools-preview"
                  ];
                }
              ))
            ];
            shellHook = ''
            '';
          };
      }
    );
}
