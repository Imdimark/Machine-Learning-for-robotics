%clearing
clear 
clc
%initialization 
twoclass = [];
labels = [];
twoclass = [];

%load MINST dataset (training set) by using the function loadMNIST
% M is the set and T the label
[M,T] = loadMNIST(0);
%disp (M(2,:));

%creating a sub-class with the only numbers 2 and 5
for x=1:length(T)
    if T(x,1) == 0  %clearing
clear 
clc
%initialization of the matrix that will hosts the class
twoclass = [];
labels = [];

%load MINST dataset (training set) by using the function loadMNIST
% M is the set and T the label
[M,T] = loadMNIST(0);
%disp (M(2,:));
twoclass = [];
y = 1;

%creating a sub-class with the only numbers 2 and 5
for x=1:length(T)
    %disp(T(x));
    
    if T(x,1) == 1  

        twoclass(end + 1, :) = M(x,:);
        labels (end +1,1) = T(x,1);
        %labels (end +1 ,1) = T(x,1);
        
    
    elsif T(x,1) == 8  

        twoclass(end + 1, :) = M(x,:);
        labels (end +1,1) = T(x,1);
        %labels (end +1 ,1) = T(x,1);
   
      end

    x = x + 1;
end
twoclass_transpose = transpose (twoclass);
 

%these functions use the classe that has been created
myAutoencoder = trainAutoencoder(twoclass_transpose,2);
myEncodedData = encode (myAutoencoder, twoclass_transpose);
myEncodedData_transposed = transpose (myEncodedData);



figure
plotcl(myEncodedData_transposed,labels)
title (['Classes: ', num2str(2),' and ',num2str(5)])
xlabel("Activation value hidden 1")
ylabel("Activation Value hidden 2")
        twoclass(end + 1, :) = M(x,:);
        labels (end +1,1) = T(x,1);  
    elseif T(x,1) == 1  
        twoclass(end + 1, :) = M(x,:);
        labels (end +1,1) = T(x,1);
    end
    x = x + 1;
end
twoclass_transpose = transpose (twoclass); 

%these functions use the classe that has been created
myAutoencoder = trainAutoencoder(twoclass_transpose,2); 
myEncodedData = encode (myAutoencoder, twoclass_transpose);
myEncodedData_transposed = transpose (myEncodedData); %converting to "observations in rows, variables in columns"

%creating figure using plotcl
figure
plotcl(myEncodedData_transposed,labels)
title (['Classes: ', num2str(8),' and ',num2str(1)])
xlabel("Activation hidden neuron 1")
ylabel("Activation hidden neuron 2")