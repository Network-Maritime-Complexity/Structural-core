mkdir('CM_networks')
networks = {'GLSN2015'};
iters = 1000;
percs = 0.1:0.1:0.5;

for i1 = 1:length(networks)
    load([networks{i1} '.mat'], 'x')
    for i2 = 1:length(percs)
        time = tic;
        fprintf('%s %.1f ... ', networks{i1}, percs(i2));
        matrices = cell(iters,1);
        parfor j = 1:iters
            matrices{j} = randomize_network(x, 'CM', percs(i2));
        end
        save(['CM_networks/' networks{i1} '_CM_perc' num2str(percs(i2)*100) '.mat'], 'matrices')
        fprintf('[%ds]\n', round(toc(time)));
    end
end

% check disconnected networks and replace
for i1 = 1:length(networks)
    load([networks{i1} '.mat'], 'x')
    for i2 = 1:length(percs)
        load(['CM_networks/' networks{i1} '_CM_perc' num2str(percs(i2)*100) '.mat'], 'matrices')
        for j = 1:iters
            while graphconncomp(matrices{j},'Directed',false) > 1
                fprintf('%s %.1f %d - DISCONNECTED\n', networks{i1}, percs(i2), j);
                matrices{j} = randomize_network(x, 'CM', percs(i2));
            end
        end
        save(['CM_networks/' networks{i1} '_CM_perc' num2str(percs(i2)*100) '.mat'], 'matrices')
    end
end