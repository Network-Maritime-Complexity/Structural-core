function plot_figure_core_sub_comm

% load sub data
load('sub.mat', 'x1')
load('sub1_params.mat', 'core', 'module')
core = logical(core);
x = x1;

% embedding
coords = coalescent_embedding(x, 'RA2', 'ISO', 'EA', 2);

% plot figure
figure('color', 'white')

plot_embedding_comm_core(x, coords, module, core)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_embedding_comm_core(x, coords, comm, core)

[~,idx] = sortrows(core, 1);
x = x(idx,idx);
coords = coords(idx,:);
comm = comm(idx);    
core = core(idx);
    
colors = [[255 0 0] ./ 255;     % red
          [255 0 255] ./ 255; 	% violet
          [42 255 128] ./ 255;  % green
          [255 140 0] ./ 255;   % orange
          [85 221 255] ./ 255;  % light blue
          [255 242 0] ./ 255;   % yellow
          [0 0 255] ./ 255];    % blue
      
colors = colors(comm,:);

hold on
radius = max(coords(:,2));
[coords_x,coords_y] = pol2cart(coords(:,1),coords(:,2));

[h1,h2] = gplot(x(~core,~core), [coords_x(~core), coords_y(~core)], 'k');
plot(h1, h2, 'Color', [0.6,0.6,0.6], 'LineWidth', 0.8)
[h1,h2] = gplot(x(core,core), [coords_x(core), coords_y(core)]);
plot(h1, h2, 'Color', 'k', 'LineWidth', 6)

scatter(coords_x(~core), coords_y(~core), 1400, colors(~core,:), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);  % none core
for i = 1:length(x)
    if core(i)
        scatter(coords_x(i), coords_y(i), 1500, 'k', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
        scatter(coords_x(i), coords_y(i), 1200, colors(i,:), 'filled', 'MarkerEdgeColor', 'none');
    end
end
xlim([-radius,radius])
ylim([-radius,radius])
axis square
axis off
