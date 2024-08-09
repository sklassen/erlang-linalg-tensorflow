-module(linalg_tf_tests). 
-import(linalg_tf,[to_tensor/1,from_tensor/1,transpose/1,inv/1,matmul/2,svd/1,diag/1]).
-include_lib("eunit/include/eunit.hrl").

% linalg_tf:from_tensor(linalg_tf:to_tensor([[1.0,2.0],[3.0,4.0]])).

transpose_1_test() ->
	?assertEqual([[8.0]],from_tensor(transpose(to_tensor([[8]])))).

transpose_2_test() ->
	?assertEqual([[1.0,3.0],[2.0,4.0]],from_tensor(transpose(to_tensor([[1.0,2.0],[3.0,4.0]])))).

transpose_3_test() ->
	?assertEqual([[1.0,4.0],[2.0,5.0],[3.0,6.0]],from_tensor(transpose(to_tensor([[1.0,2.0,3.0],[4.0,5.0,6.0]])))).

%diag_2_test() ->
%	?assertEqual([1.1,4.5],from_tensor(diag(to_tensor([[1.1,2.1],[3.1,4.5]])))).

inv_1_test() ->
	?assertEqual([[0.125]],from_tensor(inv(to_tensor([[8.0]])))).

inv_2_test()->
	?assertEqual([[1.0,2.0],[2.0,0.5]],from_tensor(inv(to_tensor([[1.0,0.5],[0.5,2.0]])))).

%inv_3_test()->
%	?assertEqual([[-1.0,-1.0,2.0],[-1.0,0.0,1.0],[2.0,1.0,-2.0]],from_tensor(inv(to_tensor([[1.0,0.0,1.0],[0.0,2.0,1.0],[1.0,1.0,1.0]])))).

matmul_1_test()->
  ?assertEqual([[6.0]], from_tensor(matmul(to_tensor([[2.0]]), to_tensor([[3.0]])))).

matmul_2_test()->
  ?assertEqual([[5.0,11.0],[11.0,25.0]], from_tensor(matmul(to_tensor([[1.0,2.0],[3.0,4.0]]), to_tensor([[1.0,3.0],[2.0,4.0]])))).

%svd_2x2_test() ->
%  A=to_tensor([[3,2],[2,3]]),
%  {U,S,Vt}=svd(A),
%  [
%   ?assertEqual([[0.707,0.707],[0.707,-0.707]],linalg:around(U,3)),
%   ?assertEqual([5, 1], linalg:around(S)),
%   ?assertEqual([[0.707,0.707],[0.707,-0.707]],linalg:around(Vt,3)),
%   ?assertEqual(A,linalg:around(matmul(matmul(U,diag(S)),Vt)))
%  ].

%svd_2x3_test() ->
%  A=[[3,2,2],[2,3,-2]],
%  {U,S,Vt}=svd(A),
%  [
%   ?assertEqual([[0.707,0.707],[0.707,-0.707]],linalg:around(U,3)),
%   ?assertEqual([5, 3], linalg:around(S)),
%   ?assertEqual([[0.707,0.707,0.0],[0.236,-0.236,0.943]],linalg:around(Vt,3)),
%   ?assertEqual(A,linalg:around(matmul(matmul(U,diag(S)),Vt)))
%  ].


%svd_3x2_test() ->
%  A=[[3,2],[2,3],[2,-2]],
%  {U,S,Vt}=svd(A),
%  [
%   ?assertEqual([[0.707,0.236],[0.707,-0.236],[0.0,0.943]],linalg:around(U,3)),
%   ?assertEqual([5, 3], linalg:around(S)),
%   ?assertEqual([[0.707,0.707],[0.707,-0.707]],linalg:around(Vt,3)),
%   ?assertEqual(A,linalg:around(matmul(matmul(U,diag(S)),Vt)))
%  ].
%
