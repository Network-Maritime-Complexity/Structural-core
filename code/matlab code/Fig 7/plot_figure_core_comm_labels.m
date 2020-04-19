function plot_figure_core_comm_labels

% load GLSN data
load('trade_GLSN.mat', 'x')
load('trade_GLSN_metadata.mat', 'core', 'module')

% embedding
coords = coalescent_embedding(x, 'RA2', 'ISO', 'EA', 2);

% plot figure
figure('color', 'white')

plot_figure_comm_core_labels(x, coords, module, core)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function plot_figure_comm_core_labels(x, coords, comm, core)

[~,idx] = sortrows(core, 1);
x = x(idx,idx);
coords = coords(idx,:);
comm = comm(idx); 
core = core(idx);
    
colors = [[102 0 150] ./ 255;     % purple
          [255 0 197] ./ 255; 	% pink
          [217 133 0] ./ 255;  % yellow
          [0 105 214] ./ 255;   % blue
          [0 0 0] ./ 255;  % black
          [232 76 0] ./ 255;   % yorange
          [63 166 35] ./ 255];    % green
      
colors = colors(comm,:);

hold on
radius = max(coords(:,2));
[coords_x,coords_y] = pol2cart(coords(:,1),coords(:,2));

[h1,h2] = gplot(x(~core,~core), [coords_x(~core), coords_y(~core)], 'k');
plot(h1, h2, 'Color', [0.8,0.8,0.8], 'LineWidth', 0.001)
[h1,h2] = gplot(x(core,core), [coords_x(core), coords_y(core)]);
plot(h1, h2, 'Color', [0.25,0.25,0.25], 'LineWidth', 2)

scatter(coords_x(~core), coords_y(~core), 250, colors(~core,:), 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);  % none core
for i = 1:length(x)
    if core(i)
        scatter(coords_x(i), coords_y(i), 500, 'k', 'filled', 'MarkerEdgeColor', 'k', 'LineWidth', 1);
        scatter(coords_x(i), coords_y(i), 250, colors(i,:), 'filled', 'MarkerEdgeColor', 'none');
    end
end
xlim([-radius,radius])
ylim([-radius,radius])
axis square
axis off
