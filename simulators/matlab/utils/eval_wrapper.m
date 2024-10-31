function fun_name = eval_wrapper(step)
y = eval(step);
global string_fun
string_fun = eval("@(i) interp1((1:500)/100, y, i, 'linear')");
fun_name = "@(t)string_fun(t)";

end
