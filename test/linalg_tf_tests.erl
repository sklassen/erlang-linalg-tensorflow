-module(linalg_tf_tests). 
-import(linalg_tf,[transpose/1]).
-include_lib("eunit/include/eunit.hrl").

transpose_1_test() ->
	?assertEqual([[8.0]],transpose([[8.0]])).

transpose_2_test() ->
	?assertEqual([[1.0,3.0],[2.0,4.0]],transpose([[1.0,2.0],[3.0,4.0]])).

transpose_3_test() ->
	?assertEqual([[1.0,4.0],[2.0,5.0],[3.0,6.0]],transpose([[1.0,2.0,3.0],[4.0,5.0,6.0]])).


