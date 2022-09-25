1.main.o : 1.main_func.cpp
	@g++ -c $< -o $@ -std=c++11

pro1 : 1.main.o
	@g++ $< -o $@

main_func: pro1
	@./pro1

##########################

2.main.o : 2.data_type.cpp
	@g++ -c $< -o $@ -std=c++11

pro2 : 2.main.o
	@g++ $< -o $@

data_type: pro2
	@./pro2
##########################
	

3.main.o : 3.pointer.cpp
	@g++ -c $< -o $@ -std=c++11

pro3 : 3.main.o
	@g++ $< -o $@

pointer : pro3
	@./pro3
##########################


4.1main.o : 4.1func.cpp
	@g++ -c $< -o $@ -std=c++11

pro4.1 : 4.1main.o
	@g++ $< -o $@

func : pro4.1
	@./pro4.1
##########################

4.2main.o : 4.2lambda.cpp
	@g++ -c $< -o $@ -std=c++11

pro4.2 : 4.2main.o
	@g++ $< -o $@

lambda : pro4.2
	@./pro4.2