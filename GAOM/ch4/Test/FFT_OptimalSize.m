%%
%Generate the prefered FFT sizes less than 4096

m = 0;
II = zeros(13*13*13*13,1);
for i=0:12
    for j = 0:12
        for k = 0:12
            for l = 0:12
                temp = 2^i * 3^j * 5^k * 7^l;
                if(mod(temp,2)==0 && temp <= 4096)
                    m = m+1;
                    II(m) = temp;
                end
            end
        end
    end
end

II = nonzeros(II);
I = sort(II)';

fileID = fopen('4096.txt','w');
fprintf(fileID,'%d, ', I);
fclose(fileID);

