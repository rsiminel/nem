{
  description = "fastnem";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-26.05";

  outputs = { self, nixpkgs }:
    let
      systems = [ "x86_64-linux" "aarch64-linux" "x86_64-darwin" "aarch64-darwin" ];
      forAllSystems = nixpkgs.lib.genAttrs systems;
    in {
      devShells = forAllSystems (system:
        let pkgs = nixpkgs.legacyPackages.${system}; in {
          default = pkgs.mkShell {
            name = "fastnem";
            packages = [
              pkgs.gcc
              pkgs.cmake
              pkgs.ninja
              pkgs.uv
              pkgs.ccls
            ];
          };
        });
    };
}
