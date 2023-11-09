%Tested on Octave version 3.2.4 configured for "i686-pc-linux-gnu"

%Backgammon functions

function board = setUpBoard()
	board = [0, 2, 0, 0, 0, 0, -5, 0, -3, 0, 0, 0, 5, -5, 0, 0, 0, 3, 0, 5, 0, 0, 0, 0, -2, 0]; %bar & points
endfunction


function [n1, n2] = roll()
	n1 = round(rand() * 5) + 1;
	n2 = round(rand() * 5) + 1;
endfunction


function board = doMove(board, wh, p, n)
	if p == 0
		return
	endif

	c = sign(board(p));		%occupier of origin point
	p2 = p + wh*n;			%destination point

	if 1 < p2 && p2 < 26		%otherwise we're bearing this piece off
		d = board(p2);

		if sign(d) == -c	%if a blot is on the destination
			if abs(d) > 1	%this should never happen
				printf("Error: destination for move is owned by opponent.\n")
			endif

			if wh == 1	%put the blot on its owner's bar
				board(26) -= 1;
			else
				board(1) += 1;
			endif

			board(p2) = 0;
		endif

		board(p2) += wh; %move piece to destination
	endif

	board(p) -= wh; %remove piece from origin
endfunction


function board = doMoves(board, wh, m)
	for i = 1 : 2 : size(m, 2)
		board = doMove(board, wh, m(1, i+1), m(1, i));
	endfor
endfunction


function v = isLegal(board, wh, p, n)
%isLegal only checks for legality of single moves, regardless of other
%possible moves
	v = true; %until proven otherwise

	%sanity checks
	if wh != -1 && wh != 1
		v = false;
		printf("Error: sanity check failed. wh = %i\n", wh)
		return
	elseif p < 1 || 26 < p
		v = false;
		printf("Error: sanity check failed. p = %i\n", p)
		return
	elseif n < 1 || 6 < n
		v = false;
		printf("Error: sanity check failed. n = %i\n", n)
		return
	endif

	c = sign(board(p));		%occupier of origin point
	p2 = p + wh*n;			%destination point
	if p2 < 1 || 26 < p2
		c2 = 0;
	else
		c2 = sign(board(p2));	%occupier of destination point
	endif

	%can only move your own pieces
	if c != wh || c == 0
		v = false;
%		printf("not yours\n")
		return
	endif

	%can only move to a point occupied by opponent if it's a blot
	if c2 == -c && abs(board(p2)) > 1
		v = false;
%		printf("occupied\n")
		return
	endif

	%must move your pieces off the bar first
	if wh == 1 && board(1) != 0 && p != 1
		v = false;
%		printf("bar\n")
		return
	elseif wh == -1 && board(26) != 0 && p != 26
		v = false;
%		printf("bar\n")
		return
	endif

	%can only bear off once all your pieces are in your home board
	if wh == 1 && 25 < p2 && any(board(1:19) > 0)
		v = false;
%		printf("home\n")
		return
	elseif wh == -1 && p2 < 2 && any(board(8:26) < 0)
		v = false;
%		printf("home\n")
		return
	endif

	%must use exact rolls to bear off unless higher points are empty
	if wh == 1 && 26 < p2 && any(board(20:p-1) > 0)
		v = false;
%		printf("exact\n")
		return
	elseif wh == -1 && p2 < 1 && any(board(p+1:7) < 0)
		v = false;
%		printf("exact\n")
		return
	endif
endfunction


function m = legalMoves1(board, wh, n)
	m = [];
	for i = 1 : 26
		if isLegal(board, wh, i, n)
			m = [m; i];
		endif
	endfor
endfunction


function m = legalMoves(board, wh, n1, n2)
	m = [];

	if n1 == n2
		m1 = legalMoves1(board, wh, n1);
		if isempty(m1)
			m = [n1, 0, n1, 0, n1, 0, n1, 0];
		endif
		for h = 1 : size(m1, 1)
			nb1 = doMove(board, wh, m1(h), n1);
			m2 = legalMoves1(nb1, wh, n1);
			if wh == 1
				m2 = m2(m2 >= m1(h));
			else
				m2 = m2(m2 <= m1(h));
			endif
			if isempty(m2)
				m = [m; [n1, m1(h), n1, 0, n1, 0, n1, 0]];
			endif

			for i = 1 : size(m2, 1)
				nb2 = doMove(nb1, wh, m2(i), n1);
				m3 = legalMoves1(nb2, wh, n1);
				if wh == 1
					m3 = m3(m3 >= m2(i));
				else
					m3 = m3(m3 <= m2(i));
				endif
				if isempty(m3)
					m = [m; [n1, m1(h), n1, m2(i), n1, 0, n1, 0]];
				endif

				for j = 1 : size(m3, 1)
					nb3 = doMove(nb2, wh, m3(j), n1);
					m4 = legalMoves1(nb3, wh, n1);
					if wh == 1
						m4 = m4(m4 >= m3(j));
					else
						m4 = m4(m4 <= m3(j));
					endif
					if isempty(m4)
						m = [m; [n1, m1(h), n1 m2(i), n1, m3(j), n1, 0]];
					endif

					s4 = size(m4);
					c = n1(ones(s4));
					nm = [c, m1(h)(ones(s4)), c, m2(i)(ones(s4)), c, m3(j)(ones(s4)), c, m4];
					m = [m; nm];
				endfor
			endfor
		endfor
	else
		m1 = legalMoves1(board, wh, n1);
		m2 = legalMoves1(board, wh, n2);

		for i = 1 : size(m1, 1)
			nb = doMove(board, wh, m1(i), n1);
			m2g1 = legalMoves1(nb, wh, n2);
			if wh == 1
				m2g1 = m2g1(m2g1 >= m1(i));
			else
				m2g1 = m2g1(m2g1 <= m1(i));
			endif
			if isempty(m2g1)
				m2g1 = [0];
			endif

			s2 = size(m2g1);
			nm = [n1(ones(s2)), m1(i)(ones(s2)), n2(ones(s2)), m2g1];
			m = [m; nm];
		endfor

		for i = 1 : size(m2, 1)
			nb = doMove(board, wh, m2(i), n2);
			m1g2 = legalMoves1(nb, wh, n1);
			if wh == 1
				m1g2 = m1g2(m1g2 >= m2(i));
			else
				m1g2 = m1g2(m1g2 <= m2(i));
			endif
			if isempty(m1g2)
				m1g2 = [0];
			endif

			s1 = size(m1g2);
			nm = [n2(ones(s1)), m2(i)(ones(s1)), n1(ones(s1)), m1g2];
			m = [m; nm];
		endfor

		if isempty(m)
			m = [n1, 0, n2, 0];
		endif
	endif

	%this deletes moves that do not use the dice as fully as possible
	tm = zeros(1, size(m, 1));
	for i = 1 : size(m, 1)
		tm(i) = sum(m(i, [1 : 2 : size(m, 2)]) .* (m(i, [2 : 2 : size(m, 2)]) != 0));
	endfor
	m = m(tm == max(tm), :);
endfunction


function b = nextBoards(board, wh, n1, n2)
	b = [];

	m = legalMoves(board, wh, n1, n2);

	for i = 1 : size(m, 1)
		nb = board;
		for j = 1 : 2 : size(m, 2)
			nb = doMove(nb, wh, m(i, j+1), m(i, j));
		endfor
		b = [b; nb];
	endfor
endfunction


function p = isWinner(board)
	pipP = sum(board(board > 0));
	pipN = sum(board(board < 0));
	if pipP > 0 && pipN < 0
		p = [0 0 0 0]';
	elseif pipP == 0 && pipN < 0
		if pipN == -15
			p = [0 0 0 1]';
		else
			p = [0 0 1 0]';
		endif
	elseif pipP > 0 && pipN == 0
		if pipP == 15
			p = [0 1 0 0]';
		else
			p = [1 0 0 0]';
		endif
	elseif pipP == 0 && pipN == 0
		printf("Error: no pieces on board!\n");
		p = [0 0 0 0]';
	endif
endfunction



%Utility functions

function board = oneSide(board, wh)
%returns the board with all pieces not belonging to player "wh" removed

	board((wh * board) < 0) = 0;
endfunction


function cBoard = encBoard(board, wh, t)
	global verbose;

	bar1 = board(26);
	bar2 = board(1);
	p1n = sum(board(board < 0)); %number of pieces for each player
	p2n = sum(board(board > 0));

	bmb = board(2:25); %board minus bar
	stats = [1, bar1, bar2, p1n, p2n, wh]; %the initial "1" provides the constant term in the learned function

	switch t
		%these first three encodings are "knowledge free"
		case 1 %54 inputs
			cBoard = [stats, oneSide(bmb, -1), oneSide(bmb, 1)]';
		case 2 %102 inputs
			cBoard = [stats, bmb == -1, bmb == 1, oneSide(bmb+1, -1), oneSide(bmb-1, 1)]';
		case 3 %150 inputs
			cBoard = [stats, bmb == -1, bmb == 1, bmb == -2, bmb == 2, oneSide(bmb+2, -1), oneSide(bmb-2, 1)]';
		case 4 %152 inputs
		%includes features about the number of primes owned by each player
		%total number of primes and size of largest prime

			%gaps are the points which are not owned by each player
			gaps1 = find([0 bmb 0] > -2);
			gaps2 = find([0 bmb 0] < 2);

			%jmp lists the size of each prime (plus 1)
			jmp1 = shift(gaps1, -1) - gaps1;
			jmp2 = shift(gaps2, -1) - gaps2;

			feat = [sum(jmp1 > 2), sum(jmp2 > 2)];

			cBoard = [stats, feat, bmb == -1, bmb == 1, bmb == -2, bmb == 2, oneSide(bmb+2, -1), oneSide(bmb-2, 1)]';
		case 5
		%this is wayyy too slow to use for training, or much else

			lm = zeros(6, 2);
			for i = 1 : 6
				lm(i, 1) = length(legalMoves1(board, -1, i));
				lm(i, 2) = length(legalMoves1(board, 1, i));
			endfor

			%these four features attempt to capture the flexibility and range of the current position for each player
			feat = [sum(lm(:, 1)), sum(lm(:, 2)), sum(lm(:, 1) .* [1:6]'), sum(lm(:, 2) .* [1:6]')];

			cBoard = [stats, feat, oneSide(bmb, -1), oneSide(bmb, 1)]';
	endswitch

	if verbose >= 3
		printf("In encBoard:\n")
		printf("Encoding type: %u\n", t)
		printf("cBoard':\n")
		cBoard'
		pause
	endif
endfunction


function [nBoard, maxnp] = chooseMove(v, w, t, board, wh, n1, n2)
	global verbose;

	nb = nextBoards(board, wh, n1, n2);

	if verbose >= 3
		printf("In chooseMove:\n")
		nb
		pause
	endif

	maxsc = -2; %max score, start with worst possible
	maxnb = 0; %index of chosen next board
	maxnp = [0 0 0 0]'; %prediction for next board

	for i = 1 : size(nb, 1)
		np = isWinner(nb(i, :));
		if np == [0 0 0 0]'
			%no confirmed outcome, so continue to predict
			np = fwdNet(v, w, encBoard(nb(i, :), -wh, t));
		endif

		sc = [-1 -2 1 2] * np * wh; %weight the probabilities by potential score
		if sc > maxsc
			maxsc = sc;
			maxnb = i;
			maxnp = np;
		endif

		if verbose >= 3
			printf("In chooseMove:\n")
			printf("i = %u of %u\n", i, size(nb, 1))
			np
			sc
			maxsc
			maxnb
			maxnp
			pause
		endif
	endfor

	nBoard = nb(maxnb, :);
endfunction


function writeNet(v, w, t, argName, argDest, num)
	global verbose;

	[dir, name, ext] = fileparts(argName);
	dname = str2double(name);
	if isdir(argDest)
		if isnan(dname)
			%if the filename is non-numeric
			if length(name) == 7 && name(1) == "~" && name(7) == "~"
				%if this is a new net with no preexisting filename
				path = fullfile(argDest, [num2str(num) ".n"]);
			else
				path = fullfile(argDest, [name "+" num2str(num) ext]);
			endif
		else
			path = fullfile(argDest, [num2str(dname + num) ext]);
		endif
	else
		path = argDest;
	endif

	save(path, "t", "v", "w");

	if verbose >= 3
		printf("In writeNet:\n")
		printf("wrote net to \"%s\"\n", path)
		pause
	endif
endfunction


function [vs, ws, ts] = loadNets(nets)
	global verbose;

	vs = {};
	ws = {};
	ts = [];
	for i = 1 : size(nets, 1)
		if length(nets{i, 1}) == 7 && nets{i, 1}(1) == "~" && nets{i, 1}(7) == "~" %TODO make this check a regex instead of this crap
			nh = str2double(nets{i, 1}(2:4));
			nt = str2double(nets{i, 1}(6));
			[nv, nw] = newNet(nh, nt);
			vs = [vs; {nv}];
			ws = [ws; {nw}];
			ts = [ts; nt];

			if verbose >= 3
				printf("In loadNets:\n")
				printf("Created network of type %u with %u hidden nodes and %u input nodes\n", nt, nh, ni)
				pause
			endif

			if nets{i, 2} == true
				writeNet(nv, nw, nt, nets{i, 1}, nets{i, 4}, 0) %TODO delete this maybe
			endif
		else
			try
				load(nets{i, 1}, "t", "v", "w")
				vs = [vs; {v}];
				ws = [ws; {w}];
				ts = [ts; t];
				if verbose >= 3
					printf("In loadNets:\n")
					printf("Loaded network %s\n", nets{i, 1})
					pause
				endif
			catch
				msg = lasterr();
				printf("%s\n", msg)
			end_try_catch
		endif
	endfor
endfunction


function [sched, params] = makeSchedule(nets, params)
	eta = params(1);
	lambda = params(2);
	games = params(3);
	mode = params(4);

	sched = [];

	%create the schedule of play
	switch mode
		case 1
			if size(nets, 1) < 1
				printf("Error: at least 1 net needed for self-play!\n")
				quit
			endif

			sched = [1 : size(nets, 1)]';
			loops = games;
		case 2
			if size(nets, 1) < 2
				printf("Error: at least 2 nets needed for versus play!\n")
				quit
			endif

			sched = nchoosek(1 : size(nets, 1), 2);
%			netc = nchoosek(1 : size(nets, 1), 2);
%			for i = 1 : size(netc, 1)
%				sched = [sched; netc(i, :)];
%			endfor

			loops = ceil(games / (size(nets, 1) - 1));
			if mod(games, size(nets, 1) - 1) != 0
				printf("Warning: Each net will play %i games instead of %i\n", loops * (size(nets, 1) - 1), games)
			endif
		case 3
			if size(nets, 1) < 1
				printf("Error: at least 1 net needed for user-play!\n")
				quit
			endif

			sched = [1 : size(nets, 1)]';
			loops = games;
			
	endswitch

	params = [eta, lambda, loops, mode];
endfunction


function r = diceRow(n, r)
	if n < 1 || 6 < n
		printf("Error: diceRow received invalid die value!")
		quit
	endif
	if r < 1 || 3 < r
		if r < 0 || 4 < r
			r = "     ";
		else
			r = " --- ";
		endif
		return
	endif
	dice = {"|   |";
		"| * |";
		"|   |";
		"|*  |";
		"|   |";
		"|  *|";
		"|*  |";
		"| * |";
		"|  *|";
		"|* *|";
		"|   |";
		"|* *|";
		"|* *|";
		"| * |";
		"|* *|";
		"|* *|";
		"|* *|";
		"|* *|"};
	r = dice{3*n - r + 1};
endfunction



%ANN functions

function [v, w] = newNet(h, t)
	v = rand(4, h+1) * 0.01 - 0.005;
	w = rand(h, length(encBoard(setUpBoard(), 1, t)) ) * 0.01 - 0.005;
endfunction


function z = sigmoid(w, x)
	x = x(:);
	z = 1 ./ (1 + exp(- w * x));
endfunction


function p = softmax(v, z)
	z = z(:);
	exps = exp(v * z);
	p = exps / sum(exps);
endfunction


function dp = diffzSoftmax(v, z)
%diffzSoftmax computes the derivative of softmax w.r.t. z (output of hidden layer)

	global verbose;

	z = z(:);
	dp = zeros(size(v));
	dxexps = zeros(size(v));

	exps = exp(v * z);
	se = sum(exps);

	for i = 1 : size(v, 1)
		dxexps(i, :) = v(i, :) * exps(i);
	endfor
	sde = sum(dxexps, 1);

	for i = 1 : size(v, 1)
		%row "i" is derivative of ith net output w.r.t. output of hidden layer
		dp(i, :) = exps(i) * (v(i, :) * (se - exps(i)) - (sde - dxexps(i, :))) / (se ^ 2);
	endfor

	if verbose >= 4
		printf("In diffzSoftmax:\n")
		dp
		pause
	endif
endfunction


function [gv, gw] = gVgW(v, w, cBoard, z, p)
%gVgW computes the gradients of each output value with respect to v and w
%this is part of the update for v and w

	global verbose;

	%"p(i) * (eye(4)(:, i) - p) * z'" computes the gradient of p(i) w.r.t. v
	for i = 1 : 4
		gv{i} = p(i) * (eye(4)(:, i) - p) * z';
	endfor

	dsm = diffzSoftmax(v, z);
	for i = 1 : 4
		gw{i} = dsm(i, 2:end)' .* z(2:end) .* (1 - z(2:end)) * cBoard';
	endfor

	if verbose >= 4
		printf("In gVgW:\n")
		gv
		gw
		pause
	endif
endfunction


function [p, z] = fwdNet(v, w, cBoard)
%forward propagation through the network defined by v and w

	global verbose;

	cBoard = cBoard(:);
	z = [1; sigmoid(w, cBoard)];
	p = softmax(v, z);

	if verbose >= 3
		printf("In fwdNet:\n")
		z
		p
		pause
	endif
endfunction



%Playing interface functions

function [v, w, win] = selfGame(v, w, t, eta, lambda, update=true)
	global verbose;
	gv = {zeros(size(v)) zeros(size(v)) zeros(size(v)) zeros(size(v))};
	gw = {zeros(size(w)) zeros(size(w)) zeros(size(w)) zeros(size(w))};

	dv = zeros(size(v));
	dw = zeros(size(w));

	n1 = 0;
	n2 = 0;
	while (n1 == n2) %can't move doubles on first move
		[n1, n2] = roll(); %rolling to see who goes first
	endwhile

	if n1 > n2
		wh = -1;
	else
		wh = 1;
	endif

	board = setUpBoard();

	cBoard = encBoard(board, wh, t);
	win = [0 0 0 0]';
	plies = 0;

	while win == [0 0 0 0]'
		if verbose >= 2
			printf("In selfGame:\n")
			wh
			board
			plies
			printf("Roll: %u, %u\n", n1, n2)
			pause
		endif

		[p, z] = fwdNet(v, w, cBoard);

		if plies < 500
			[nBoard, maxnp] = chooseMove(v, w, t, board, wh, n1, n2);
		else
			if verbose >= 2
				printf("In selfGame:\n")
				printf("Switching to random moves after 500 plies\n")
			endif

			%to speed up slow games and get the net out of its rut
			nb = nextBoards(board, wh, n1, n2);
			i = round(rand()*(size(nb, 1) - 1) + 1);

			nBoard = nb(i, :);
			maxnp = fwdNet(v, w, encBoard(nBoard, wh, t));
		endif

		%compute updates to v and w
		if update == true
			[dgv, dgw] = gVgW(v, w, cBoard, z, p);
			diff = maxnp - p;

			for i = 1 : 4
				gv{i} = gv{i} * lambda + dgv{i};
				gw{i} = gw{i} * lambda + dgw{i};
				dv += diff(i) * gv{i};
				dw += diff(i) * gw{i};
			endfor
		endif

		wh = -wh;
		board = nBoard; %update board
		cBoard = encBoard(nBoard, wh, t);
		[n1, n2] = roll();

		plies++;
		win = isWinner(nBoard);
	endwhile

	if update == true
		v += eta * dv;
		w += eta * dw;
	endif
endfunction


function [v1, w1, v2, w2, win] = versusGame(v1, w1, t1, v2, w2, t2, eta, lambda, u1=true, u2=true)
%same as selfGame, except given networks take turns choosing moves
%both networks learn from the entire game

	global verbose;

	gv1 = {zeros(size(v1)) zeros(size(v1)) zeros(size(v1)) zeros(size(v1))};
	gw1 = {zeros(size(w1)) zeros(size(w1)) zeros(size(w1)) zeros(size(w1))};
	gv2 = {zeros(size(v2)) zeros(size(v2)) zeros(size(v2)) zeros(size(v2))};
	gw2 = {zeros(size(w2)) zeros(size(w2)) zeros(size(w2)) zeros(size(w2))};

	dv1 = zeros(size(v1));
	dw1 = zeros(size(w1));
	dv2 = zeros(size(v2));
	dw2 = zeros(size(w2));

	n1 = 0;
	n2 = 0;
	while (n1 == n2) %can't move doubles on first move
		[n1, n2] = roll(); %rolling to see who goes first
	endwhile

	if n1 > n2
		wh = -1;
	else
		wh = 1;
	endif

	board = setUpBoard();

	cBoard1 = encBoard(board, wh, t1);
	cBoard2 = encBoard(board, wh, t2);
	win = [0 0 0 0]';
	plies = 0;

	while win == [0 0 0 0]'
		if verbose >= 2
			printf("In versusGame:\n")
			wh
			board
			plies
			printf("Roll: %u, %u\n", n1, n2)
			pause
		endif

		if wh == -1
			[p, z] = fwdNet(v1, w1, cBoard1);
		elseif wh == 1
			[p, z] = fwdNet(v2, w2, cBoard2);
		endif

		if wh == -1
			[nBoard, maxnp] = chooseMove(v1, w1, t1, board, wh, n1, n2);
		elseif wh == 1
			[nBoard, maxnp] = chooseMove(v2, w2, t2, board, wh, n1, n2);
		endif

		%compute updates to networks
		if wh == -1
			if u2 == true
				[po, zo] = fwdNet(v2, w2, cBoard2);
				maxnpo = fwdNet(v2, w2, encBoard(nBoard, -wh, t2));
				[dgv2, dgw2] = gVgW(v2, w2, cBoard2, zo, po);
				diff2 = maxnpo - po;
			endif

			if u1 == true
				[dgv1, dgw1] = gVgW(v1, w1, cBoard1, z, p);
				diff1 = maxnp - p;
			endif
		elseif wh == 1
			if u1 == true
				[po, zo] = fwdNet(v1, w1, cBoard1);
				maxnpo = fwdNet(v1, w1, encBoard(nBoard, -wh, t1));
				[dgv1, dgw1] = gVgW(v1, w1, cBoard1, zo, po);
				diff1 = maxnpo - po;
			endif

			if u2 == true
				[dgv2, dgw2] = gVgW(v2, w2, cBoard2, z, p);
				diff2 = maxnp - p;
			endif
		endif


		for i = 1 : 4
			if u1 == true
				gv1{i} = gv1{i} * lambda + dgv1{i};
				gw1{i} = gw1{i} * lambda + dgw1{i};
				dv1 += diff1(i) * gv1{i};
				dw1 += diff1(i) * gw1{i};
			endif
			if u2 == true
				gv2{i} = gv2{i} * lambda + dgv2{i};
				gw2{i} = gw2{i} * lambda + dgw2{i};
				dv2 += diff2(i) * gv2{i};
				dw2 += diff2(i) * gw2{i};
			endif
		endfor

		%update current game state
		wh = -wh;
		board = nBoard;
		cBoard1 = encBoard(nBoard, wh, t1);
		cBoard2 = encBoard(nBoard, wh, t2);
		[n1, n2] = roll();

		plies++;
		win = isWinner(nBoard);
	endwhile

	if u1 == true
		v1 += eta * dv1;
		w1 += eta * dw1;
	endif
	if u2 == true
		v2 += eta * dv2;
		w2 += eta * dw2;
	endif
endfunction


function [v, w, win] = userGame(v, w, t, eta, lambda, update=true)
	global verbose;
	gv = {zeros(size(v)) zeros(size(v)) zeros(size(v)) zeros(size(v))};
	gw = {zeros(size(w)) zeros(size(w)) zeros(size(w)) zeros(size(w))};

	dv = zeros(size(v));
	dw = zeros(size(w));

	n1 = 0;
	n2 = 0;
	while (n1 == n2) %can't move doubles on first move
		[n1, n2] = roll(); %rolling to see who goes first
	endwhile

	if n1 > n2
		wh = -1;
	else
		wh = 1;
	endif

	if verbose == 0
		%establish who's who
		%the logic here works, trust me
		fflush(stdout);
		if round(rand()) == 0
			p1ok = yes_or_no("You are player 1, OK? ");
			p2ok = true;
		else
			p2ok = yes_or_no("You are player 2, OK? ");
			p1ok = false;
		endif

		if p1ok || !p2ok
			uu = -1;
			printf("Your pieces are \"@\"\n")
		else
			uu = 1;
			printf("Your pieces are \"+\"\n")
		endif

		if uu == wh
			printf("\nYou move first\n")
		else
			printf("\nThe network moves first\n")
		endif
	else
		if round(rand()) == 0
			printf("You are player 1 (\"@\")\n")
			uu = -1;
		else
			printf("You are player 2 (\"+\")\n")
			uu = 1;
		endif
	endif

	board = setUpBoard();

	cBoard = encBoard(board, wh, t);
	win = [0 0 0 0]';
	plies = 0;

	while win == [0 0 0 0]'
		dispBoard(board, n1, n2);

		[p, z] = fwdNet(v, w, cBoard);

		if wh == uu
			punm = legalMoves(board, uu, n1, n2);

			printf("You rolled: %u and %u\n\n", n1, n2)
			printf("Your available moves:\n")

			if n1 == n2
				if ! any(punm([2 4 6 8]))
					printf("\nNo moves available!\n")
					punm = [];
				else
					for i = 1 : size(punm, 1)
						os = [punm(i, [2 4 6 8])] - 1;
						ds = os + wh * n1;
						printf("%2u:  %2u/%2u,  %2u/%2u,  %2u/%2u,  %2u/%2u\n", i, os(1), ds(1), os(2), ds(2), os(3), ds(3), os(4), ds(4))
					endfor
				endif
				printf("\n")
			else
				if ! any(punm([2 4]))
					printf("\nNo moves available!\n")
					punm = [];
				else
					for i = 1 : size(punm, 1)
						os = [punm(i, [2 4])] - 1;
						ds = os + wh * punm(i, [1 3]);
						printf("%2u:  %2u/%2u,  %2u/%2u  (%1u, %1u)\n", i, os(1), ds(1), os(2), ds(2), punm(i, 1), punm(i, 3))
					endfor
				endif
				printf("\n")
			endif

			if ! isempty(punm)
				unm = NaN;
				valid = false;
				while ! valid
					unm = input("Enter the number of your choice: ", "s");
					unm = round(str2double(unm));
					if isnan(unm) || ! isscalar(unm) || unm < 1 || size(punm, 1) < unm
						printf("Invalid option!\n")
					else
						valid = true;
					endif
				endwhile

				nBoard = doMoves(board, uu, punm(unm, :));
			else
				nBoard = board;
			endif

			maxnp = fwdNet(v, w, encBoard(nBoard, -wh, t));
		else
			%pause here so the player can see the result of his/her move
			%and to give the impression that the computer is thinking really hard
			if verbose == 0
				pause(3)
			endif
			[nBoard, maxnp] = chooseMove(v, w, t, board, wh, n1, n2);
		endif

		%compute updates to v and w
		if update == true
			[dgv, dgw] = gVgW(v, w, cBoard, z, p);
			diff = maxnp - p;

			for i = 1 : 4
				gv{i} = gv{i} * lambda + dgv{i};
				gw{i} = gw{i} * lambda + dgw{i};
				dv += diff(i) * gv{i};
				dw += diff(i) * gw{i};
			endfor
		endif

		wh = -wh;
		board = nBoard; %update board
		cBoard = encBoard(nBoard, wh, t);
		[n1, n2] = roll();

		plies++;
		win = isWinner(nBoard);
	endwhile

	if update == true
		v += eta * dv;
		w += eta * dw;
	endif

	if uu == -1 %runUGSched expects the user to be player 2
		win = win([3 4 1 2]);
	endif
endfunction


function runSGSched(nets, sched, params)
	global verbose;
	eta = params(1);
	lambda = params(2);
	loops = params(3);

	[vs, ws, ts] = loadNets(nets);

	if size(nets, 1) < 1
		printf("Error: required nets failed to load!\n")
		quit
	endif

	pc = zeros(size(nets, 1)); %play count
	sc = zeros(size(nets, 1), 2); %scores

	if verbose == 0
		printf("Games played each:  ")
	endif

	for i = 1 : loops
		for j = 1 : size(sched, 1)
			[v, w, win] = selfGame(vs{j}, ws{j}, ts(j), eta, lambda, nets{j, 2});
			if verbose >= 1
				printf("In runSGSched:\n")
				printf("player = %u\n", j)
				printf("win = [ %u %u %u %u ]\n", win(1), win(2), win(3), win(4))
				pause
			endif

			sc(j, 1) += win(1) + 2 * win(2);
			sc(j, 2) += win(3) + 2 * win(4);

			%update play count
			pc(j, j) += 1;

			if verbose >= 1
				printf("In runSGSched:\n")
				sc
				pc
				pause
			endif

			%update and write networks
			if nets{j, 2} == true
				vs{j} = v;
				ws{j} = w;
				if mod(pc(j), nets{j, 3}) == 0 %if it's time to write out updated nets
					writeNet(v, w, ts(j), nets{j, 1}, nets{j, 4}, pc(j))
				endif
			endif
		endfor

		if verbose == 0
			for r = 1 : length(num2str(i-1))
				printf("\b")
			endfor
			printf("%i", i)
		endif
	endfor

	printf("\n\n    Scores\n--------------\n")
	for i = 1 : size(nets, 1)
		printf("Net: %s\t\tp1: %4u  p2: %4u\n", nets{i, 1}, sc(i, 1), sc(i, 2))
	endfor
endfunction


function runVGSched(nets, sched, params)
	global verbose;
	eta = params(1);
	lambda = params(2);
	loops = params(3);

	[vs, ws, ts] = loadNets(nets);

	if size(nets, 1) < 2
		printf("Error: required nets failed to load!\n")
		quit
	endif

	pc = zeros(size(nets, 1)); %play count
	sc = zeros(size(nets, 1), 1); %scores

	if verbose == 0
		printf("\n     Net:  ")
		for i = 1 : min(19, size(nets, 1)) %TODO make this adapt to the actual width of the screen
			printf("%5u", i)
		endfor
		printf("\nComplete  |  Scores\n")
	endif

	for i = 1 : loops
		for j = 1 : size(sched, 1)
			p1 = sched(j, 1);
			p2 = sched(j, 2);

			if mod(pc(p1, p2) + pc(p2, p1), 2) == 0 %this gives each net a fair chance, since they improve unevenly between playing each side
				[v1, w1, v2, w2, win] = versusGame(vs{p1}, ws{p1}, ts(p1), vs{p2}, ws{p2}, ts(p2), eta, lambda, nets{p1, 2}, nets{p2, 2});
				%update play count
				pc(p1, p2) += 1;
			else
				[v2, w2, v1, w1, win] = versusGame(vs{p2}, ws{p2}, ts(p2), vs{p1}, ws{p1}, ts(p1), eta, lambda, nets{p2, 2}, nets{p1, 2});
				win = win([3 4 1 2]');
				pc(p2, p1) += 1;
			endif

			if verbose >= 1
				printf("In runVGSched:\n")
				printf("p1 = %u\np2 = %u\n", p1, p2)
				printf("win = [ %u %u %u %u ]\n", win(1), win(2), win(3), win(4))
				pause
			endif

			%update scores
			sc(p1) += win(1) + 2 * win(2);
			sc(p2) += win(3) + 2 * win(4);

			if verbose >= 1
				printf("In runVGSched:\n")
				sc
				pc
				pause
			endif

			%print status update
			if verbose == 0
				per = (i - 1 + (j / size(sched, 1))) / loops; %percent complete
				per = floor(100 * per);
				printf("\r    %3u%%   ", per)
				for k = 1 : min(19, size(nets, 1))
					printf("%5u", sc(k))
				endfor
			endif

			%update and write networks
			if nets{p1, 2} == true %if player 1's net should be updated
				vs{p1} = v1;
				ws{p1} = w1;
				pc1 = sum(pc(p1, :)) + sum(pc(:, p1));
				if mod(pc1, nets{p1, 3}) == 0 %if it's time
					writeNet(v1, w1, ts(p1), nets{p1, 1}, nets{p1, 4}, pc1)
				endif
			endif
			if nets{p2, 2} == true %player 2
				vs{p2} = v2;
				ws{p2} = w2;
				pc2 = sum(pc(p2, :)) + sum(pc(:, p2));
				if mod(pc2, nets{p2, 3}) == 0
					writeNet(v2, w2, ts(p2), nets{p2, 1}, nets{p2, 4}, pc2)
				endif
			endif
		endfor
	endfor

	printf("\n\n    Scores\n--------------\n")
	for i = 1 : size(nets, 1)
		printf("Net: %s\t\tScore: %u\n", nets{i, 1}, sc(i))
	endfor
endfunction


function runUGSched(nets, sched, params)
	global verbose
	eta = params(1);
	lambda = params(2);
	loops = params(3);

	[vs, ws, ts] = loadNets(nets);

	if size(nets, 1) < 1
		printf("Error: required nets failed to load!\n")
		quit
	endif

	pc = zeros(size(nets, 1)); %play count
	sc = zeros(size(nets, 1), 2); %scores

	for i = 1 : loops
		for j = 1 : size(sched, 1)
			if verbose == 0
				printf("Next opponent: %s\n", nets{j, 1})
				brave = yes_or_no("Continue? ");
				if ! brave
					quit
				endif
			endif

			[v, w, win] = userGame(vs{j}, ws{j}, ts(j), eta, lambda, nets{j, 2});
			sc(j, 1) += win(1) + 2 * win(2);
			sc(j, 2) += win(3) + 2 * win(4);

			%update play count
			pc(j, j) += 1;

			if verbose >= 1
				printf("In runUGSched:\n")
				sc
				pc
				pause
			endif

			printf("\n\n    Scores\n--------------\n")
			for i = 1 : size(nets, 1)
				printf("Net: %s\t\tnet: %4u  user: %4u\n", nets{i, 1}, sc(i, 1), sc(i, 2))
			endfor

			%update and write networks
			if nets{j, 2} == true
				vs{j} = v;
				ws{j} = w;
				if mod(pc(j), nets{j, 3}) == 0 %if it's time to write out updated nets
					writeNet(v, w, ts(j), nets{j, 1}, nets{j, 4}, pc(j))
				endif
			endif
		endfor
	endfor
endfunction


function runSchedule(nets, sched, params)
	eta = params(1);
	lambda = params(2);
	loops = params(3);
	mode = params(4);

	%check for correct schedule
	if size(sched, 1) < 1
		printf("Error: schedule failed to initialize!\n")
		quit
	endif

	%print training schedule
	printf("\nNets to use:\n")
	for i = 1 : size(nets, 1)
		if nets{i, 2} == true
			%attach the correct suffix to the update interval number
			switch mod(nets{i, 3}, 10)
				case 1
					suf = "st";
				case 2
					suf = "nd";
				case 3
					suf = "rd";
				otherwise
					suf = "th";
			endswitch
			if 10 < mod(nets{i, 3}, 100) && mod(nets{i, 3}, 100) < 20
				suf = "th";
			endif
			printf("%i: %s (every %u%s net saved to %s)\n", i, nets{i, 1}, nets{i, 3}, suf, nets{i, 4})
		else
			printf("%i: %s (Not updated)\n", i, nets{i, 1})
		endif
	endfor

	printf("\nLearning rate: %g\nTemporal credit discount rate: %g\n", params(1), params(2))

	switch mode
		case 1 %self play
			runSGSched(nets, sched, params)
		case 2 %versus play
			runVGSched(nets, sched, params)
		case 3 %user play
			runUGSched(nets, sched, params)
		otherwise
			printf("Error in parseArgs!\n")
			quit
	endswitch
endfunction



%User interface functions

function helpMsg()
printf("\nUSAGE\n\n")

printf("octave tdgammon.m [options]\n\n\n")


printf(" OPTIONS\n")
printf("---------\n\n")

printf("-n PATH\t\tLoad a network from the given file or directory\n")
printf("\t\tIf PATH is a file, it must contain scalar \"t\", and\n")
printf("\t\tmatrices \"v\" and \"w\"\n\n")
printf("\t\tIf PATH is a directory, all files in it must contain\n")
printf("\t\t\"t\" \"v\" and \"w\"\n\n")
printf("\t\tIf PATH is in the following format, a new network will be\n")
printf("\t\tgenerated and initialized with random values\n\n")

printf("\t\t~hhh,t~\t\tWhere \"hhh\" is the number of hidden nodes\n")
printf("\t\t\t\tand \"t\" is the type of input encoding.\n")
printf("\t\t\t\tValid values for \"t\" are 1, 2, 3, or 4.\n\n")

printf("-w\t\tTurn on updates\n\n")

printf("-nw\t\tTurn off updates\n\n")

printf("-i NUM\t\tSet network write-outs to occur every NUM games\n")
printf("\t\tValid values: [1 - Inf]  Default: 100\n\n")

printf("-o PATH\t\tWrite saved networks to PATH\n\n")

printf("-e NUM\t\tSet learning rate \"eta\" to NUM\n")
printf("\t\tValid values: [0 - Inf]  Default: 0.1\n\n")

printf("-l NUM\t\tSet temporal credit discount rate \"lambda\" to NUM\n")
printf("\t\tValid values: [0 - 1]  Default: 0.7\n\n")

printf("-g NUM\t\tSchedule each network to play NUM games\n\n")
printf("\t\tValid values: [1 - Inf]  Default: 1\n\n")

printf("-s\t\tSelf-play mode: each network plays itself\n\n")

printf("-v\t\tVersus mode: each network plays each other network\n\n")

printf("-u\t\tUser-play mode: each network in turn will play the user\n\n")

printf("-c NUM\t\tCopious output: increase debugging output to level NUM\n")
printf("\t\tValid values: [0, 1, 2, 3, 4]  Default: 0\n\n")

printf("Options \"-w\", \"-nw\", \"-i\", and \"-o\" can be given multiple times.\n")
printf("Each time they are given, they apply to the network most recently\n")
printf("specified (with \"-n\").  If no nets have yet been specified, they apply\n")
printf("to all nets, until overridden.\n\n")

printf("The filenames for saved networks are automatically generated as follows:\n")
printf("If the filename (minus extension) is numeric, it will be incremented by\n")
printf("the number of games it has played during this invocation of the program.\n\n")
printf("If the filename is non-numeric, it will be saved as \"NAME+NUM.EXT\"\n")
printf("where NAME is the original filename, NUM is the number of games played, and\n")
printf("EXT is the original file extension.\n\n")
printf("If the network was a new network (see option \"-n\" above) the filename will\n")
printf("be \"NUM.n\" where NUM is the number of games played.\n\n\n")


printf(" EXAMPLES\n")
printf("----------\n\n")

printf("octave tdgammon.m -n 1000.n -i 10 -o 1000.n -n 2000.n -nw -v -e .01 -g 100\n\n")

printf("This will play \"1000.n\" against \"2000.n\" for 100 games.\n")
printf("\"1000.n\" will be overwritten every 10 games. \"2000.n\" will not be\n")
printf("updated.  The learning rate is 0.01.\n\n\n")


printf("octave tdgammon.m -n ver1 -i 10 -o ver2 -s -g 200\n\n")

printf("Every net saved in directory \"ver1\" will be played against itself 200\n")
printf("times.  Every 10 games, each one will be written out to directory \"ver2\"\n")
printf("with a new file name.\n\n\n")


printf("octave tdgammon.m -n ~040,1~ -i 1000 -s -g 10000\n\n")
printf("A new network will be generated with 40 hidden nodes and using the no. 1\n")
printf("board encoding.  It will play 10000 games and write out every 1000th game\n")
printf("to the current directory.\n")
endfunction


function [nets, params] = parseArgs()
	global verbose;

	args = argv();
	isarg = false;
	nets = {"" NaN NaN ""};

	%params = [eta, lambda, num games, mode]
	%defaults
	params = [0.1, 0.7, 1, 1];

	i = 1;
	nextnet = 1;
	while i <= length(args)
		if length(args{i}) >= 2 && args{i}(1) == "-"
			isarg = true;
			switch args{i}(2:end) %TODO add argument to update a network in place, overwrite the same file
				case "n" %next argument is a network to use
					nets{nextnet++, 1} = args{++i};
					%add a placeholder for the next net's options
					nets = [nets; [{""} nets(nextnet-1, 2:3) {""}]];
				case "w" %this and subsequent nets will be updated
					if nextnet > 1 && isnan(nets{nextnet-1, 2})
						nets{nextnet-1, 2} = true;
					else
						nets{nextnet, 2} = true;
					endif
				case "nw" %this and subsequent nets will not be updated
					if nextnet > 1 && isnan(nets{nextnet-1, 2})
						nets{nextnet-1, 2} = false;
					else
						nets{nextnet, 2} = false;
					endif
				case "i" %interval games between updates
					val = round(str2double(args{++i}));

					if isnan(val) || ! isscalar(val) || val < 1
						printf("Error: invalid value for option \"-i\"\n")
						helpMsg();
						quit
					endif

					if nextnet > 1 && isnan(nets{nextnet-1, 3})
						nets{nextnet-1, 3} = val;
					else
						nets{nextnet, 3} = val;
					endif
				case "o" %where to output saved networks
					if nextnet > 1 && isempty(nets{nextnet-1, 4})
						nets{nextnet-1, 4} = args{++i};
					else
						nets{nextnet, 4} = args{++i};
					endif

				case "e" %learning rate "eta"
					val = str2double(args{++i});

					if isnan(val) || ! isscalar(val) || val < 0
						printf("Error: invalid value for option \"-e\"\n")
						helpMsg();
						quit
					endif
					params(1) = val;
				case "l" %temporal discount rate "lambda"
					val = str2double(args{++i});

					if isnan(val) || ! isscalar(val) || val < 0 || 1 < val
						printf("Error: invalid value for option \"-l\"\n")
						helpMsg();
						quit
					endif
					params(2) = val;
				case "g" %number of games each net will play
					val = round(str2double(args{++i}));

					if isnan(val) || ! isscalar(val) || val < 1
						printf("Error: invalid value for option \"-g\"\n")
						helpMsg();
						quit
					endif
					params(3) = val;
				case "s" %self play mode
					params(4) = 1;
				case "v" %versus play mode
					params(4) = 2;
				case "u" %user play mode
					params(4) = 3;
				case "c" %copious output
					val = round(str2double(args{++i}));

					if isnan(val) || ! isscalar(val) || val < 0
						printf("Error: invalid value for option \"-c\"\n")
						helpMsg();
						quit
					endif
					verbose = val;
				otherwise
					printf("Option %s is not recognized!\n", args{i})
			endswitch
		else
			printf("Argument %s is not recognized!\n", args{i})
		endif
		i++;
	endwhile

	if isarg == false
		helpMsg();
		quit
	endif

	%remove the empty placeholder at the end of "nets"
	nets = nets(1:end-1, :);

	%list all nets in given dirs
	i = 1;
	while i <= size(nets, 1)
		if isdir(nets{i, 1})
			l = readdir(nets{i});
			ins = {};
			for j = 1 : length(l)
				path = fullfile(nets{i, 1}, l{j});
				if ! isdir(path)
					ins = [ins; [{path} nets(i, 2:4)]];
				endif
			endfor

			nets = [nets(1:i-1, :); ins; nets(i+1:end, :)];
		endif
		i++;
	endwhile

	%make sure all nets have good options set
	for i = 1 : size(nets, 1)
		if ! isnan(nets{i, 2}) || ! isnan(nets{i, 3}) || ! isempty(nets{i, 4})
			if isnan(nets{i, 2})
				nets{i, 2} = true;
			endif
			if isnan(nets{i, 3})
				nets{i, 3} = 100;
			endif
			if isempty(nets{i, 4})
				if length(nets{i, 1}) == 7 && nets{i, 1}(1) == "~" && nets{i, 1}(7) == "~"
					nets{i, 4} = ".";
				else
					dir = fileparts(nets{i, 1});
					nets{i, 4} = dir;
				endif
			endif
		else
			nets(i, 2:3) = {false, 0};
		endif
	endfor
endfunction


function dispBoard(board, n1, n2)
	pBoard = oneSide(board, 1);
	nBoard = oneSide(board, -1);
	nBoard = abs(nBoard);

	maxp = max([pBoard nBoard]);

	printf("\n")
	for h = max(maxp, 5) : -1 : 1
		for i = 1 : length(board)
			if pBoard(i) >= h
				printf(" + ")
			elseif nBoard(i) >= h
				printf(" @ ")
			else
				printf("   ")
			endif
		endfor

		cols = str2double(getenv("COLUMNS"));
		if cols >= 92 %if there's screen room
			d1r = diceRow(n1, h-1);
			d2r = diceRow(n2, h-1);
			printf("  %s  %s\n", d1r, d2r)
		else
			printf("\n")
		endif
	endfor
	printf("BAR 1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 BAR\n")
	printf("       .     .     .|    .     .     .|    .     .     .|    .     .     .\n")
endfunction



%%%Main code%%%

global verbose = 0
[nets, params] = parseArgs();
[sched, params] = makeSchedule(nets, params); %modifies "params"
runSchedule(nets, sched, params);
quit

%For testing

%        1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26
%board = [0  0  0  0  2  0  0  1  0  0 -1  0  0  0  0  1  0  0  0 -1  0  0  0  0  0  0];
%board = setUpBoard();
%dispBoard(board, 3, 4)
%load([pwd "/../data/nets/40h_4/" "22000" ".n"], "v", "w")
%fwdNet(v, w, encBoard(board, -1, 4))
%fwdNet(v, w, encBoard(board, 1, 4))
%nb = chooseMove(v, w, 4, board, -1, 3, 4);
%dispBoard(nb, 3, 4)
%fwdNet(v, w, encBoard(nb, -1, 4))
%quit
