-module(linalg_tf).
-export([version/0,to_tensor/1,from_tensor/1,transpose/1,echo/1]).

-on_load(init/0).

init() ->
  Directory=filename:dirname(code:which(linalg_tf)),
	erlang:load_nif(Directory++"/../priv/librustf", 0).

version() -> 
        exit(nif_library_not_loaded).

transpose(_) -> 
        exit(nif_library_not_loaded).

echo(_) -> 
        exit(nif_library_not_loaded).

to_tensor(_) -> 
        exit(nif_library_not_loaded).

from_tensor(_) -> 
        exit(nif_library_not_loaded).
