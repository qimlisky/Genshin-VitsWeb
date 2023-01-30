(function () {
	const e = document.createElement('link').relList;
	if (e && e.supports && e.supports('modulepreload')) return;
	for (const s of document.querySelectorAll('link[rel="modulepreload"]'))
		i(s);
	new MutationObserver((s) => {
		for (const o of s)
			if (o.type === 'childList')
				for (const r of o.addedNodes)
					r.tagName === 'LINK' && r.rel === 'modulepreload' && i(r);
	}).observe(document, { childList: !0, subtree: !0 });
	function n(s) {
		const o = {};
		return (
			s.integrity && (o.integrity = s.integrity),
			s.referrerpolicy && (o.referrerPolicy = s.referrerpolicy),
			s.crossorigin === 'use-credentials'
				? (o.credentials = 'include')
				: s.crossorigin === 'anonymous'
				? (o.credentials = 'omit')
				: (o.credentials = 'same-origin'),
			o
		);
	}
	function i(s) {
		if (s.ep) return;
		s.ep = !0;
		const o = n(s);
		fetch(s.href, o);
	}
})();
function Pi(t, e) {
	const n = Object.create(null),
		i = t.split(',');
	for (let s = 0; s < i.length; s++) n[i[s]] = !0;
	return e ? (s) => !!n[s.toLowerCase()] : (s) => !!n[s];
}
const Pr =
		'itemscope,allowfullscreen,formnovalidate,ismap,nomodule,novalidate,readonly',
	Dr = Pi(Pr);
function mo(t) {
	return !!t || t === '';
}
function Di(t) {
	if (P(t)) {
		const e = {};
		for (let n = 0; n < t.length; n++) {
			const i = t[n],
				s = pt(i) ? Nr(i) : Di(i);
			if (s) for (const o in s) e[o] = s[o];
		}
		return e;
	} else {
		if (pt(t)) return t;
		if (tt(t)) return t;
	}
}
const Fr = /;(?![^(]*\))/g,
	Hr = /:(.+)/;
function Nr(t) {
	const e = {};
	return (
		t.split(Fr).forEach((n) => {
			if (n) {
				const i = n.split(Hr);
				i.length > 1 && (e[i[0].trim()] = i[1].trim());
			}
		}),
		e
	);
}
function Fi(t) {
	let e = '';
	if (pt(t)) e = t;
	else if (P(t))
		for (let n = 0; n < t.length; n++) {
			const i = Fi(t[n]);
			i && (e += i + ' ');
		}
	else if (tt(t)) for (const n in t) t[n] && (e += n + ' ');
	return e.trim();
}
function Rr(t, e) {
	if (t.length !== e.length) return !1;
	let n = !0;
	for (let i = 0; n && i < t.length; i++) n = Wn(t[i], e[i]);
	return n;
}
function Wn(t, e) {
	if (t === e) return !0;
	let n = ps(t),
		i = ps(e);
	if (n || i) return n && i ? t.getTime() === e.getTime() : !1;
	if (((n = nn(t)), (i = nn(e)), n || i)) return t === e;
	if (((n = P(t)), (i = P(e)), n || i)) return n && i ? Rr(t, e) : !1;
	if (((n = tt(t)), (i = tt(e)), n || i)) {
		if (!n || !i) return !1;
		const s = Object.keys(t).length,
			o = Object.keys(e).length;
		if (s !== o) return !1;
		for (const r in t) {
			const l = t.hasOwnProperty(r),
				c = e.hasOwnProperty(r);
			if ((l && !c) || (!l && c) || !Wn(t[r], e[r])) return !1;
		}
	}
	return String(t) === String(e);
}
function jr(t, e) {
	return t.findIndex((n) => Wn(n, e));
}
const Je = (t) =>
		pt(t)
			? t
			: t == null
			? ''
			: P(t) || (tt(t) && (t.toString === vo || !j(t.toString)))
			? JSON.stringify(t, go, 2)
			: String(t),
	go = (t, e) =>
		e && e.__v_isRef
			? go(t, e.value)
			: ke(e)
			? {
					[`Map(${e.size})`]: [...e.entries()].reduce(
						(n, [i, s]) => ((n[`${i} =>`] = s), n),
						{}
					),
			  }
			: Kn(e)
			? { [`Set(${e.size})`]: [...e.values()] }
			: tt(e) && !P(e) && !_o(e)
			? String(e)
			: e,
	G = {},
	Me = [],
	Ft = () => {},
	Br = () => !1,
	Ur = /^on[^a-z]/,
	qn = (t) => Ur.test(t),
	Hi = (t) => t.startsWith('onUpdate:'),
	gt = Object.assign,
	Ni = (t, e) => {
		const n = t.indexOf(e);
		n > -1 && t.splice(n, 1);
	},
	Wr = Object.prototype.hasOwnProperty,
	Y = (t, e) => Wr.call(t, e),
	P = Array.isArray,
	ke = (t) => dn(t) === '[object Map]',
	Kn = (t) => dn(t) === '[object Set]',
	ps = (t) => dn(t) === '[object Date]',
	j = (t) => typeof t == 'function',
	pt = (t) => typeof t == 'string',
	nn = (t) => typeof t == 'symbol',
	tt = (t) => t !== null && typeof t == 'object',
	bo = (t) => tt(t) && j(t.then) && j(t.catch),
	vo = Object.prototype.toString,
	dn = (t) => vo.call(t),
	qr = (t) => dn(t).slice(8, -1),
	_o = (t) => dn(t) === '[object Object]',
	Ri = (t) =>
		pt(t) && t !== 'NaN' && t[0] !== '-' && '' + parseInt(t, 10) === t,
	Tn = Pi(
		',key,ref,ref_for,ref_key,onVnodeBeforeMount,onVnodeMounted,onVnodeBeforeUpdate,onVnodeUpdated,onVnodeBeforeUnmount,onVnodeUnmounted'
	),
	zn = (t) => {
		const e = Object.create(null);
		return (n) => e[n] || (e[n] = t(n));
	},
	Kr = /-(\w)/g,
	De = zn((t) => t.replace(Kr, (e, n) => (n ? n.toUpperCase() : ''))),
	zr = /\B([A-Z])/g,
	je = zn((t) => t.replace(zr, '-$1').toLowerCase()),
	$o = zn((t) => t.charAt(0).toUpperCase() + t.slice(1)),
	oi = zn((t) => (t ? `on${$o(t)}` : '')),
	sn = (t, e) => !Object.is(t, e),
	On = (t, e) => {
		for (let n = 0; n < t.length; n++) t[n](e);
	},
	kn = (t, e, n) => {
		Object.defineProperty(t, e, {
			configurable: !0,
			enumerable: !1,
			value: n,
		});
	},
	on = (t) => {
		const e = parseFloat(t);
		return isNaN(e) ? t : e;
	};
let ms;
const Vr = () =>
	ms ||
	(ms =
		typeof globalThis < 'u'
			? globalThis
			: typeof self < 'u'
			? self
			: typeof window < 'u'
			? window
			: typeof global < 'u'
			? global
			: {});
let jt;
class Yr {
	constructor(e = !1) {
		(this.active = !0),
			(this.effects = []),
			(this.cleanups = []),
			!e &&
				jt &&
				((this.parent = jt),
				(this.index = (jt.scopes || (jt.scopes = [])).push(this) - 1));
	}
	run(e) {
		if (this.active) {
			const n = jt;
			try {
				return (jt = this), e();
			} finally {
				jt = n;
			}
		}
	}
	on() {
		jt = this;
	}
	off() {
		jt = this.parent;
	}
	stop(e) {
		if (this.active) {
			let n, i;
			for (n = 0, i = this.effects.length; n < i; n++)
				this.effects[n].stop();
			for (n = 0, i = this.cleanups.length; n < i; n++)
				this.cleanups[n]();
			if (this.scopes)
				for (n = 0, i = this.scopes.length; n < i; n++)
					this.scopes[n].stop(!0);
			if (this.parent && !e) {
				const s = this.parent.scopes.pop();
				s &&
					s !== this &&
					((this.parent.scopes[this.index] = s),
					(s.index = this.index));
			}
			this.active = !1;
		}
	}
}
function Xr(t, e = jt) {
	e && e.active && e.effects.push(t);
}
const ji = (t) => {
		const e = new Set(t);
		return (e.w = 0), (e.n = 0), e;
	},
	xo = (t) => (t.w & oe) > 0,
	yo = (t) => (t.n & oe) > 0,
	Jr = ({ deps: t }) => {
		if (t.length) for (let e = 0; e < t.length; e++) t[e].w |= oe;
	},
	Qr = (t) => {
		const { deps: e } = t;
		if (e.length) {
			let n = 0;
			for (let i = 0; i < e.length; i++) {
				const s = e[i];
				xo(s) && !yo(s) ? s.delete(t) : (e[n++] = s),
					(s.w &= ~oe),
					(s.n &= ~oe);
			}
			e.length = n;
		}
	},
	mi = new WeakMap();
let Qe = 0,
	oe = 1;
const gi = 30;
let Lt;
const Ee = Symbol(''),
	bi = Symbol('');
class Bi {
	constructor(e, n = null, i) {
		(this.fn = e),
			(this.scheduler = n),
			(this.active = !0),
			(this.deps = []),
			(this.parent = void 0),
			Xr(this, i);
	}
	run() {
		if (!this.active) return this.fn();
		let e = Lt,
			n = ie;
		for (; e; ) {
			if (e === this) return;
			e = e.parent;
		}
		try {
			return (
				(this.parent = Lt),
				(Lt = this),
				(ie = !0),
				(oe = 1 << ++Qe),
				Qe <= gi ? Jr(this) : gs(this),
				this.fn()
			);
		} finally {
			Qe <= gi && Qr(this),
				(oe = 1 << --Qe),
				(Lt = this.parent),
				(ie = n),
				(this.parent = void 0),
				this.deferStop && this.stop();
		}
	}
	stop() {
		Lt === this
			? (this.deferStop = !0)
			: this.active &&
			  (gs(this), this.onStop && this.onStop(), (this.active = !1));
	}
}
function gs(t) {
	const { deps: e } = t;
	if (e.length) {
		for (let n = 0; n < e.length; n++) e[n].delete(t);
		e.length = 0;
	}
}
let ie = !0;
const Co = [];
function Be() {
	Co.push(ie), (ie = !1);
}
function Ue() {
	const t = Co.pop();
	ie = t === void 0 ? !0 : t;
}
function Ot(t, e, n) {
	if (ie && Lt) {
		let i = mi.get(t);
		i || mi.set(t, (i = new Map()));
		let s = i.get(n);
		s || i.set(n, (s = ji())), wo(s);
	}
}
function wo(t, e) {
	let n = !1;
	Qe <= gi ? yo(t) || ((t.n |= oe), (n = !xo(t))) : (n = !t.has(Lt)),
		n && (t.add(Lt), Lt.deps.push(t));
}
function Qt(t, e, n, i, s, o) {
	const r = mi.get(t);
	if (!r) return;
	let l = [];
	if (e === 'clear') l = [...r.values()];
	else if (n === 'length' && P(t))
		r.forEach((c, f) => {
			(f === 'length' || f >= i) && l.push(c);
		});
	else
		switch ((n !== void 0 && l.push(r.get(n)), e)) {
			case 'add':
				P(t)
					? Ri(n) && l.push(r.get('length'))
					: (l.push(r.get(Ee)), ke(t) && l.push(r.get(bi)));
				break;
			case 'delete':
				P(t) || (l.push(r.get(Ee)), ke(t) && l.push(r.get(bi)));
				break;
			case 'set':
				ke(t) && l.push(r.get(Ee));
				break;
		}
	if (l.length === 1) l[0] && vi(l[0]);
	else {
		const c = [];
		for (const f of l) f && c.push(...f);
		vi(ji(c));
	}
}
function vi(t, e) {
	const n = P(t) ? t : [...t];
	for (const i of n) i.computed && bs(i);
	for (const i of n) i.computed || bs(i);
}
function bs(t, e) {
	(t !== Lt || t.allowRecurse) && (t.scheduler ? t.scheduler() : t.run());
}
const Zr = Pi('__proto__,__v_isRef,__isVue'),
	Eo = new Set(
		Object.getOwnPropertyNames(Symbol)
			.filter((t) => t !== 'arguments' && t !== 'caller')
			.map((t) => Symbol[t])
			.filter(nn)
	),
	Gr = Ui(),
	tl = Ui(!1, !0),
	el = Ui(!0),
	vs = nl();
function nl() {
	const t = {};
	return (
		['includes', 'indexOf', 'lastIndexOf'].forEach((e) => {
			t[e] = function (...n) {
				const i = J(this);
				for (let o = 0, r = this.length; o < r; o++)
					Ot(i, 'get', o + '');
				const s = i[e](...n);
				return s === -1 || s === !1 ? i[e](...n.map(J)) : s;
			};
		}),
		['push', 'pop', 'shift', 'unshift', 'splice'].forEach((e) => {
			t[e] = function (...n) {
				Be();
				const i = J(this)[e].apply(this, n);
				return Ue(), i;
			};
		}),
		t
	);
}
function Ui(t = !1, e = !1) {
	return function (i, s, o) {
		if (s === '__v_isReactive') return !t;
		if (s === '__v_isReadonly') return t;
		if (s === '__v_isShallow') return e;
		if (s === '__v_raw' && o === (t ? (e ? vl : Io) : e ? So : Ao).get(i))
			return i;
		const r = P(i);
		if (!t && r && Y(vs, s)) return Reflect.get(vs, s, o);
		const l = Reflect.get(i, s, o);
		return (nn(s) ? Eo.has(s) : Zr(s)) || (t || Ot(i, 'get', s), e)
			? l
			: rt(l)
			? r && Ri(s)
				? l
				: l.value
			: tt(l)
			? t
				? Mo(l)
				: Ki(l)
			: l;
	};
}
const il = To(),
	sl = To(!0);
function To(t = !1) {
	return function (n, i, s, o) {
		let r = n[i];
		if (Fe(r) && rt(r) && !rt(s)) return !1;
		if (
			!t &&
			(!Ln(s) && !Fe(s) && ((r = J(r)), (s = J(s))),
			!P(n) && rt(r) && !rt(s))
		)
			return (r.value = s), !0;
		const l = P(n) && Ri(i) ? Number(i) < n.length : Y(n, i),
			c = Reflect.set(n, i, s, o);
		return (
			n === J(o) &&
				(l ? sn(s, r) && Qt(n, 'set', i, s) : Qt(n, 'add', i, s)),
			c
		);
	};
}
function ol(t, e) {
	const n = Y(t, e);
	t[e];
	const i = Reflect.deleteProperty(t, e);
	return i && n && Qt(t, 'delete', e, void 0), i;
}
function rl(t, e) {
	const n = Reflect.has(t, e);
	return (!nn(e) || !Eo.has(e)) && Ot(t, 'has', e), n;
}
function ll(t) {
	return Ot(t, 'iterate', P(t) ? 'length' : Ee), Reflect.ownKeys(t);
}
const Oo = { get: Gr, set: il, deleteProperty: ol, has: rl, ownKeys: ll },
	cl = {
		get: el,
		set(t, e) {
			return !0;
		},
		deleteProperty(t, e) {
			return !0;
		},
	},
	al = gt({}, Oo, { get: tl, set: sl }),
	Wi = (t) => t,
	Vn = (t) => Reflect.getPrototypeOf(t);
function gn(t, e, n = !1, i = !1) {
	t = t.__v_raw;
	const s = J(t),
		o = J(e);
	n || (e !== o && Ot(s, 'get', e), Ot(s, 'get', o));
	const { has: r } = Vn(s),
		l = i ? Wi : n ? Vi : rn;
	if (r.call(s, e)) return l(t.get(e));
	if (r.call(s, o)) return l(t.get(o));
	t !== s && t.get(e);
}
function bn(t, e = !1) {
	const n = this.__v_raw,
		i = J(n),
		s = J(t);
	return (
		e || (t !== s && Ot(i, 'has', t), Ot(i, 'has', s)),
		t === s ? n.has(t) : n.has(t) || n.has(s)
	);
}
function vn(t, e = !1) {
	return (
		(t = t.__v_raw),
		!e && Ot(J(t), 'iterate', Ee),
		Reflect.get(t, 'size', t)
	);
}
function _s(t) {
	t = J(t);
	const e = J(this);
	return Vn(e).has.call(e, t) || (e.add(t), Qt(e, 'add', t, t)), this;
}
function $s(t, e) {
	e = J(e);
	const n = J(this),
		{ has: i, get: s } = Vn(n);
	let o = i.call(n, t);
	o || ((t = J(t)), (o = i.call(n, t)));
	const r = s.call(n, t);
	return (
		n.set(t, e),
		o ? sn(e, r) && Qt(n, 'set', t, e) : Qt(n, 'add', t, e),
		this
	);
}
function xs(t) {
	const e = J(this),
		{ has: n, get: i } = Vn(e);
	let s = n.call(e, t);
	s || ((t = J(t)), (s = n.call(e, t))), i && i.call(e, t);
	const o = e.delete(t);
	return s && Qt(e, 'delete', t, void 0), o;
}
function ys() {
	const t = J(this),
		e = t.size !== 0,
		n = t.clear();
	return e && Qt(t, 'clear', void 0, void 0), n;
}
function _n(t, e) {
	return function (i, s) {
		const o = this,
			r = o.__v_raw,
			l = J(r),
			c = e ? Wi : t ? Vi : rn;
		return (
			!t && Ot(l, 'iterate', Ee),
			r.forEach((f, h) => i.call(s, c(f), c(h), o))
		);
	};
}
function $n(t, e, n) {
	return function (...i) {
		const s = this.__v_raw,
			o = J(s),
			r = ke(o),
			l = t === 'entries' || (t === Symbol.iterator && r),
			c = t === 'keys' && r,
			f = s[t](...i),
			h = n ? Wi : e ? Vi : rn;
		return (
			!e && Ot(o, 'iterate', c ? bi : Ee),
			{
				next() {
					const { value: m, done: p } = f.next();
					return p
						? { value: m, done: p }
						: { value: l ? [h(m[0]), h(m[1])] : h(m), done: p };
				},
				[Symbol.iterator]() {
					return this;
				},
			}
		);
	};
}
function Gt(t) {
	return function (...e) {
		return t === 'delete' ? !1 : this;
	};
}
function ul() {
	const t = {
			get(o) {
				return gn(this, o);
			},
			get size() {
				return vn(this);
			},
			has: bn,
			add: _s,
			set: $s,
			delete: xs,
			clear: ys,
			forEach: _n(!1, !1),
		},
		e = {
			get(o) {
				return gn(this, o, !1, !0);
			},
			get size() {
				return vn(this);
			},
			has: bn,
			add: _s,
			set: $s,
			delete: xs,
			clear: ys,
			forEach: _n(!1, !0),
		},
		n = {
			get(o) {
				return gn(this, o, !0);
			},
			get size() {
				return vn(this, !0);
			},
			has(o) {
				return bn.call(this, o, !0);
			},
			add: Gt('add'),
			set: Gt('set'),
			delete: Gt('delete'),
			clear: Gt('clear'),
			forEach: _n(!0, !1),
		},
		i = {
			get(o) {
				return gn(this, o, !0, !0);
			},
			get size() {
				return vn(this, !0);
			},
			has(o) {
				return bn.call(this, o, !0);
			},
			add: Gt('add'),
			set: Gt('set'),
			delete: Gt('delete'),
			clear: Gt('clear'),
			forEach: _n(!0, !0),
		};
	return (
		['keys', 'values', 'entries', Symbol.iterator].forEach((o) => {
			(t[o] = $n(o, !1, !1)),
				(n[o] = $n(o, !0, !1)),
				(e[o] = $n(o, !1, !0)),
				(i[o] = $n(o, !0, !0));
		}),
		[t, n, e, i]
	);
}
const [fl, dl, hl, pl] = ul();
function qi(t, e) {
	const n = e ? (t ? pl : hl) : t ? dl : fl;
	return (i, s, o) =>
		s === '__v_isReactive'
			? !t
			: s === '__v_isReadonly'
			? t
			: s === '__v_raw'
			? i
			: Reflect.get(Y(n, s) && s in i ? n : i, s, o);
}
const ml = { get: qi(!1, !1) },
	gl = { get: qi(!1, !0) },
	bl = { get: qi(!0, !1) },
	Ao = new WeakMap(),
	So = new WeakMap(),
	Io = new WeakMap(),
	vl = new WeakMap();
function _l(t) {
	switch (t) {
		case 'Object':
		case 'Array':
			return 1;
		case 'Map':
		case 'Set':
		case 'WeakMap':
		case 'WeakSet':
			return 2;
		default:
			return 0;
	}
}
function $l(t) {
	return t.__v_skip || !Object.isExtensible(t) ? 0 : _l(qr(t));
}
function Ki(t) {
	return Fe(t) ? t : zi(t, !1, Oo, ml, Ao);
}
function xl(t) {
	return zi(t, !1, al, gl, So);
}
function Mo(t) {
	return zi(t, !0, cl, bl, Io);
}
function zi(t, e, n, i, s) {
	if (!tt(t) || (t.__v_raw && !(e && t.__v_isReactive))) return t;
	const o = s.get(t);
	if (o) return o;
	const r = $l(t);
	if (r === 0) return t;
	const l = new Proxy(t, r === 2 ? i : n);
	return s.set(t, l), l;
}
function Le(t) {
	return Fe(t) ? Le(t.__v_raw) : !!(t && t.__v_isReactive);
}
function Fe(t) {
	return !!(t && t.__v_isReadonly);
}
function Ln(t) {
	return !!(t && t.__v_isShallow);
}
function ko(t) {
	return Le(t) || Fe(t);
}
function J(t) {
	const e = t && t.__v_raw;
	return e ? J(e) : t;
}
function Lo(t) {
	return kn(t, '__v_skip', !0), t;
}
const rn = (t) => (tt(t) ? Ki(t) : t),
	Vi = (t) => (tt(t) ? Mo(t) : t);
function Po(t) {
	ie && Lt && ((t = J(t)), wo(t.dep || (t.dep = ji())));
}
function Do(t, e) {
	(t = J(t)), t.dep && vi(t.dep);
}
function rt(t) {
	return !!(t && t.__v_isRef === !0);
}
function ae(t) {
	return yl(t, !1);
}
function yl(t, e) {
	return rt(t) ? t : new Cl(t, e);
}
class Cl {
	constructor(e, n) {
		(this.__v_isShallow = n),
			(this.dep = void 0),
			(this.__v_isRef = !0),
			(this._rawValue = n ? e : J(e)),
			(this._value = n ? e : rn(e));
	}
	get value() {
		return Po(this), this._value;
	}
	set value(e) {
		const n = this.__v_isShallow || Ln(e) || Fe(e);
		(e = n ? e : J(e)),
			sn(e, this._rawValue) &&
				((this._rawValue = e), (this._value = n ? e : rn(e)), Do(this));
	}
}
function Tt(t) {
	return rt(t) ? t.value : t;
}
const wl = {
	get: (t, e, n) => Tt(Reflect.get(t, e, n)),
	set: (t, e, n, i) => {
		const s = t[e];
		return rt(s) && !rt(n) ? ((s.value = n), !0) : Reflect.set(t, e, n, i);
	},
};
function Fo(t) {
	return Le(t) ? t : new Proxy(t, wl);
}
var Ho;
class El {
	constructor(e, n, i, s) {
		(this._setter = n),
			(this.dep = void 0),
			(this.__v_isRef = !0),
			(this[Ho] = !1),
			(this._dirty = !0),
			(this.effect = new Bi(e, () => {
				this._dirty || ((this._dirty = !0), Do(this));
			})),
			(this.effect.computed = this),
			(this.effect.active = this._cacheable = !s),
			(this.__v_isReadonly = i);
	}
	get value() {
		const e = J(this);
		return (
			Po(e),
			(e._dirty || !e._cacheable) &&
				((e._dirty = !1), (e._value = e.effect.run())),
			e._value
		);
	}
	set value(e) {
		this._setter(e);
	}
}
Ho = '__v_isReadonly';
function Tl(t, e, n = !1) {
	let i, s;
	const o = j(t);
	return (
		o ? ((i = t), (s = Ft)) : ((i = t.get), (s = t.set)),
		new El(i, s, o || !s, n)
	);
}
function se(t, e, n, i) {
	let s;
	try {
		s = i ? t(...i) : t();
	} catch (o) {
		Yn(o, e, n);
	}
	return s;
}
function St(t, e, n, i) {
	if (j(t)) {
		const o = se(t, e, n, i);
		return (
			o &&
				bo(o) &&
				o.catch((r) => {
					Yn(r, e, n);
				}),
			o
		);
	}
	const s = [];
	for (let o = 0; o < t.length; o++) s.push(St(t[o], e, n, i));
	return s;
}
function Yn(t, e, n, i = !0) {
	const s = e ? e.vnode : null;
	if (e) {
		let o = e.parent;
		const r = e.proxy,
			l = n;
		for (; o; ) {
			const f = o.ec;
			if (f) {
				for (let h = 0; h < f.length; h++)
					if (f[h](t, r, l) === !1) return;
			}
			o = o.parent;
		}
		const c = e.appContext.config.errorHandler;
		if (c) {
			se(c, null, 10, [t, r, l]);
			return;
		}
	}
	Ol(t, n, s, i);
}
function Ol(t, e, n, i = !0) {
	console.error(t);
}
let ln = !1,
	_i = !1;
const vt = [];
let Ut = 0;
const Pe = [];
let Vt = null,
	ve = 0;
const No = Promise.resolve();
let Yi = null;
function Al(t) {
	const e = Yi || No;
	return t ? e.then(this ? t.bind(this) : t) : e;
}
function Sl(t) {
	let e = Ut + 1,
		n = vt.length;
	for (; e < n; ) {
		const i = (e + n) >>> 1;
		cn(vt[i]) < t ? (e = i + 1) : (n = i);
	}
	return e;
}
function Xi(t) {
	(!vt.length || !vt.includes(t, ln && t.allowRecurse ? Ut + 1 : Ut)) &&
		(t.id == null ? vt.push(t) : vt.splice(Sl(t.id), 0, t), Ro());
}
function Ro() {
	!ln && !_i && ((_i = !0), (Yi = No.then(Bo)));
}
function Il(t) {
	const e = vt.indexOf(t);
	e > Ut && vt.splice(e, 1);
}
function Ml(t) {
	P(t)
		? Pe.push(...t)
		: (!Vt || !Vt.includes(t, t.allowRecurse ? ve + 1 : ve)) && Pe.push(t),
		Ro();
}
function Cs(t, e = ln ? Ut + 1 : 0) {
	for (; e < vt.length; e++) {
		const n = vt[e];
		n && n.pre && (vt.splice(e, 1), e--, n());
	}
}
function jo(t) {
	if (Pe.length) {
		const e = [...new Set(Pe)];
		if (((Pe.length = 0), Vt)) {
			Vt.push(...e);
			return;
		}
		for (
			Vt = e, Vt.sort((n, i) => cn(n) - cn(i)), ve = 0;
			ve < Vt.length;
			ve++
		)
			Vt[ve]();
		(Vt = null), (ve = 0);
	}
}
const cn = (t) => (t.id == null ? 1 / 0 : t.id),
	kl = (t, e) => {
		const n = cn(t) - cn(e);
		if (n === 0) {
			if (t.pre && !e.pre) return -1;
			if (e.pre && !t.pre) return 1;
		}
		return n;
	};
function Bo(t) {
	(_i = !1), (ln = !0), vt.sort(kl);
	const e = Ft;
	try {
		for (Ut = 0; Ut < vt.length; Ut++) {
			const n = vt[Ut];
			n && n.active !== !1 && se(n, null, 14);
		}
	} finally {
		(Ut = 0),
			(vt.length = 0),
			jo(),
			(ln = !1),
			(Yi = null),
			(vt.length || Pe.length) && Bo();
	}
}
function Ll(t, e, ...n) {
	if (t.isUnmounted) return;
	const i = t.vnode.props || G;
	let s = n;
	const o = e.startsWith('update:'),
		r = o && e.slice(7);
	if (r && r in i) {
		const h = `${r === 'modelValue' ? 'model' : r}Modifiers`,
			{ number: m, trim: p } = i[h] || G;
		p && (s = n.map((C) => C.trim())), m && (s = n.map(on));
	}
	let l,
		c = i[(l = oi(e))] || i[(l = oi(De(e)))];
	!c && o && (c = i[(l = oi(je(e)))]), c && St(c, t, 6, s);
	const f = i[l + 'Once'];
	if (f) {
		if (!t.emitted) t.emitted = {};
		else if (t.emitted[l]) return;
		(t.emitted[l] = !0), St(f, t, 6, s);
	}
}
function Uo(t, e, n = !1) {
	const i = e.emitsCache,
		s = i.get(t);
	if (s !== void 0) return s;
	const o = t.emits;
	let r = {},
		l = !1;
	if (!j(t)) {
		const c = (f) => {
			const h = Uo(f, e, !0);
			h && ((l = !0), gt(r, h));
		};
		!n && e.mixins.length && e.mixins.forEach(c),
			t.extends && c(t.extends),
			t.mixins && t.mixins.forEach(c);
	}
	return !o && !l
		? (tt(t) && i.set(t, null), null)
		: (P(o) ? o.forEach((c) => (r[c] = null)) : gt(r, o),
		  tt(t) && i.set(t, r),
		  r);
}
function Xn(t, e) {
	return !t || !qn(e)
		? !1
		: ((e = e.slice(2).replace(/Once$/, '')),
		  Y(t, e[0].toLowerCase() + e.slice(1)) || Y(t, je(e)) || Y(t, e));
}
let Pt = null,
	Wo = null;
function Pn(t) {
	const e = Pt;
	return (Pt = t), (Wo = (t && t.type.__scopeId) || null), e;
}
function qo(t, e = Pt, n) {
	if (!e || t._n) return t;
	const i = (...s) => {
		i._d && Ls(-1);
		const o = Pn(e),
			r = t(...s);
		return Pn(o), i._d && Ls(1), r;
	};
	return (i._n = !0), (i._c = !0), (i._d = !0), i;
}
function ri(t) {
	const {
		type: e,
		vnode: n,
		proxy: i,
		withProxy: s,
		props: o,
		propsOptions: [r],
		slots: l,
		attrs: c,
		emit: f,
		render: h,
		renderCache: m,
		data: p,
		setupState: C,
		ctx: k,
		inheritAttrs: O,
	} = t;
	let F, L;
	const q = Pn(t);
	try {
		if (n.shapeFlag & 4) {
			const V = s || i;
			(F = Bt(h.call(V, V, m, o, C, p, k))), (L = c);
		} else {
			const V = e;
			(F = Bt(
				V.length > 1
					? V(o, { attrs: c, slots: l, emit: f })
					: V(o, null)
			)),
				(L = e.props ? c : Pl(c));
		}
	} catch (V) {
		(Ze.length = 0), Yn(V, t, 1), (F = xt(Xt));
	}
	let D = F;
	if (L && O !== !1) {
		const V = Object.keys(L),
			{ shapeFlag: H } = D;
		V.length &&
			H & 7 &&
			(r && V.some(Hi) && (L = Dl(L, r)), (D = re(D, L)));
	}
	return (
		n.dirs &&
			((D = re(D)), (D.dirs = D.dirs ? D.dirs.concat(n.dirs) : n.dirs)),
		n.transition && (D.transition = n.transition),
		(F = D),
		Pn(q),
		F
	);
}
const Pl = (t) => {
		let e;
		for (const n in t)
			(n === 'class' || n === 'style' || qn(n)) &&
				((e || (e = {}))[n] = t[n]);
		return e;
	},
	Dl = (t, e) => {
		const n = {};
		for (const i in t) (!Hi(i) || !(i.slice(9) in e)) && (n[i] = t[i]);
		return n;
	};
function Fl(t, e, n) {
	const { props: i, children: s, component: o } = t,
		{ props: r, children: l, patchFlag: c } = e,
		f = o.emitsOptions;
	if (e.dirs || e.transition) return !0;
	if (n && c >= 0) {
		if (c & 1024) return !0;
		if (c & 16) return i ? ws(i, r, f) : !!r;
		if (c & 8) {
			const h = e.dynamicProps;
			for (let m = 0; m < h.length; m++) {
				const p = h[m];
				if (r[p] !== i[p] && !Xn(f, p)) return !0;
			}
		}
	} else
		return (s || l) && (!l || !l.$stable)
			? !0
			: i === r
			? !1
			: i
			? r
				? ws(i, r, f)
				: !0
			: !!r;
	return !1;
}
function ws(t, e, n) {
	const i = Object.keys(e);
	if (i.length !== Object.keys(t).length) return !0;
	for (let s = 0; s < i.length; s++) {
		const o = i[s];
		if (e[o] !== t[o] && !Xn(n, o)) return !0;
	}
	return !1;
}
function Hl({ vnode: t, parent: e }, n) {
	for (; e && e.subTree === t; ) ((t = e.vnode).el = n), (e = e.parent);
}
const Nl = (t) => t.__isSuspense;
function Rl(t, e) {
	e && e.pendingBranch
		? P(t)
			? e.effects.push(...t)
			: e.effects.push(t)
		: Ml(t);
}
function jl(t, e) {
	if (mt) {
		let n = mt.provides;
		const i = mt.parent && mt.parent.provides;
		i === n && (n = mt.provides = Object.create(i)), (n[t] = e);
	}
}
function li(t, e, n = !1) {
	const i = mt || Pt;
	if (i) {
		const s =
			i.parent == null
				? i.vnode.appContext && i.vnode.appContext.provides
				: i.parent.provides;
		if (s && t in s) return s[t];
		if (arguments.length > 1) return n && j(e) ? e.call(i.proxy) : e;
	}
}
const Es = {};
function ci(t, e, n) {
	return Ko(t, e, n);
}
function Ko(
	t,
	e,
	{ immediate: n, deep: i, flush: s, onTrack: o, onTrigger: r } = G
) {
	const l = mt;
	let c,
		f = !1,
		h = !1;
	if (
		(rt(t)
			? ((c = () => t.value), (f = Ln(t)))
			: Le(t)
			? ((c = () => t), (i = !0))
			: P(t)
			? ((h = !0),
			  (f = t.some((L) => Le(L) || Ln(L))),
			  (c = () =>
					t.map((L) => {
						if (rt(L)) return L.value;
						if (Le(L)) return ye(L);
						if (j(L)) return se(L, l, 2);
					})))
			: j(t)
			? e
				? (c = () => se(t, l, 2))
				: (c = () => {
						if (!(l && l.isUnmounted))
							return m && m(), St(t, l, 3, [p]);
				  })
			: (c = Ft),
		e && i)
	) {
		const L = c;
		c = () => ye(L());
	}
	let m,
		p = (L) => {
			m = F.onStop = () => {
				se(L, l, 4);
			};
		};
	if (un)
		return (
			(p = Ft), e ? n && St(e, l, 3, [c(), h ? [] : void 0, p]) : c(), Ft
		);
	let C = h ? [] : Es;
	const k = () => {
		if (!!F.active)
			if (e) {
				const L = F.run();
				(i || f || (h ? L.some((q, D) => sn(q, C[D])) : sn(L, C))) &&
					(m && m(),
					St(e, l, 3, [L, C === Es ? void 0 : C, p]),
					(C = L));
			} else F.run();
	};
	k.allowRecurse = !!e;
	let O;
	s === 'sync'
		? (O = k)
		: s === 'post'
		? (O = () => yt(k, l && l.suspense))
		: ((k.pre = !0), l && (k.id = l.uid), (O = () => Xi(k)));
	const F = new Bi(c, O);
	return (
		e
			? n
				? k()
				: (C = F.run())
			: s === 'post'
			? yt(F.run.bind(F), l && l.suspense)
			: F.run(),
		() => {
			F.stop(), l && l.scope && Ni(l.scope.effects, F);
		}
	);
}
function Bl(t, e, n) {
	const i = this.proxy,
		s = pt(t) ? (t.includes('.') ? zo(i, t) : () => i[t]) : t.bind(i, i);
	let o;
	j(e) ? (o = e) : ((o = e.handler), (n = e));
	const r = mt;
	He(this);
	const l = Ko(s, o.bind(i), n);
	return r ? He(r) : Te(), l;
}
function zo(t, e) {
	const n = e.split('.');
	return () => {
		let i = t;
		for (let s = 0; s < n.length && i; s++) i = i[n[s]];
		return i;
	};
}
function ye(t, e) {
	if (!tt(t) || t.__v_skip || ((e = e || new Set()), e.has(t))) return t;
	if ((e.add(t), rt(t))) ye(t.value, e);
	else if (P(t)) for (let n = 0; n < t.length; n++) ye(t[n], e);
	else if (Kn(t) || ke(t))
		t.forEach((n) => {
			ye(n, e);
		});
	else if (_o(t)) for (const n in t) ye(t[n], e);
	return t;
}
function Ul() {
	const t = {
		isMounted: !1,
		isLeaving: !1,
		isUnmounting: !1,
		leavingVNodes: new Map(),
	};
	return (
		Qo(() => {
			t.isMounted = !0;
		}),
		Zo(() => {
			t.isUnmounting = !0;
		}),
		t
	);
}
const At = [Function, Array],
	Wl = {
		name: 'BaseTransition',
		props: {
			mode: String,
			appear: Boolean,
			persisted: Boolean,
			onBeforeEnter: At,
			onEnter: At,
			onAfterEnter: At,
			onEnterCancelled: At,
			onBeforeLeave: At,
			onLeave: At,
			onAfterLeave: At,
			onLeaveCancelled: At,
			onBeforeAppear: At,
			onAppear: At,
			onAfterAppear: At,
			onAppearCancelled: At,
		},
		setup(t, { slots: e }) {
			const n = Sc(),
				i = Ul();
			let s;
			return () => {
				const o = e.default && Xo(e.default(), !0);
				if (!o || !o.length) return;
				let r = o[0];
				if (o.length > 1) {
					for (const O of o)
						if (O.type !== Xt) {
							r = O;
							break;
						}
				}
				const l = J(t),
					{ mode: c } = l;
				if (i.isLeaving) return ai(r);
				const f = Ts(r);
				if (!f) return ai(r);
				const h = $i(f, l, i, n);
				xi(f, h);
				const m = n.subTree,
					p = m && Ts(m);
				let C = !1;
				const { getTransitionKey: k } = f.type;
				if (k) {
					const O = k();
					s === void 0 ? (s = O) : O !== s && ((s = O), (C = !0));
				}
				if (p && p.type !== Xt && (!_e(f, p) || C)) {
					const O = $i(p, l, i, n);
					if ((xi(p, O), c === 'out-in'))
						return (
							(i.isLeaving = !0),
							(O.afterLeave = () => {
								(i.isLeaving = !1), n.update();
							}),
							ai(r)
						);
					c === 'in-out' &&
						f.type !== Xt &&
						(O.delayLeave = (F, L, q) => {
							const D = Yo(i, p);
							(D[String(p.key)] = p),
								(F._leaveCb = () => {
									L(),
										(F._leaveCb = void 0),
										delete h.delayedLeave;
								}),
								(h.delayedLeave = q);
						});
				}
				return r;
			};
		},
	},
	Vo = Wl;
function Yo(t, e) {
	const { leavingVNodes: n } = t;
	let i = n.get(e.type);
	return i || ((i = Object.create(null)), n.set(e.type, i)), i;
}
function $i(t, e, n, i) {
	const {
			appear: s,
			mode: o,
			persisted: r = !1,
			onBeforeEnter: l,
			onEnter: c,
			onAfterEnter: f,
			onEnterCancelled: h,
			onBeforeLeave: m,
			onLeave: p,
			onAfterLeave: C,
			onLeaveCancelled: k,
			onBeforeAppear: O,
			onAppear: F,
			onAfterAppear: L,
			onAppearCancelled: q,
		} = e,
		D = String(t.key),
		V = Yo(n, t),
		H = (x, et) => {
			x && St(x, i, 9, et);
		},
		lt = (x, et) => {
			const U = et[1];
			H(x, et),
				P(x)
					? x.every((Q) => Q.length <= 1) && U()
					: x.length <= 1 && U();
		},
		B = {
			mode: o,
			persisted: r,
			beforeEnter(x) {
				let et = l;
				if (!n.isMounted)
					if (s) et = O || l;
					else return;
				x._leaveCb && x._leaveCb(!0);
				const U = V[D];
				U && _e(t, U) && U.el._leaveCb && U.el._leaveCb(), H(et, [x]);
			},
			enter(x) {
				let et = c,
					U = f,
					Q = h;
				if (!n.isMounted)
					if (s) (et = F || c), (U = L || f), (Q = q || h);
					else return;
				let T = !1;
				const st = (x._enterCb = (ft) => {
					T ||
						((T = !0),
						ft ? H(Q, [x]) : H(U, [x]),
						B.delayedLeave && B.delayedLeave(),
						(x._enterCb = void 0));
				});
				et ? lt(et, [x, st]) : st();
			},
			leave(x, et) {
				const U = String(t.key);
				if ((x._enterCb && x._enterCb(!0), n.isUnmounting)) return et();
				H(m, [x]);
				let Q = !1;
				const T = (x._leaveCb = (st) => {
					Q ||
						((Q = !0),
						et(),
						st ? H(k, [x]) : H(C, [x]),
						(x._leaveCb = void 0),
						V[U] === t && delete V[U]);
				});
				(V[U] = t), p ? lt(p, [x, T]) : T();
			},
			clone(x) {
				return $i(x, e, n, i);
			},
		};
	return B;
}
function ai(t) {
	if (Jn(t)) return (t = re(t)), (t.children = null), t;
}
function Ts(t) {
	return Jn(t) ? (t.children ? t.children[0] : void 0) : t;
}
function xi(t, e) {
	t.shapeFlag & 6 && t.component
		? xi(t.component.subTree, e)
		: t.shapeFlag & 128
		? ((t.ssContent.transition = e.clone(t.ssContent)),
		  (t.ssFallback.transition = e.clone(t.ssFallback)))
		: (t.transition = e);
}
function Xo(t, e = !1, n) {
	let i = [],
		s = 0;
	for (let o = 0; o < t.length; o++) {
		let r = t[o];
		const l =
			n == null ? r.key : String(n) + String(r.key != null ? r.key : o);
		r.type === kt
			? (r.patchFlag & 128 && s++, (i = i.concat(Xo(r.children, e, l))))
			: (e || r.type !== Xt) && i.push(l != null ? re(r, { key: l }) : r);
	}
	if (s > 1) for (let o = 0; o < i.length; o++) i[o].patchFlag = -2;
	return i;
}
const An = (t) => !!t.type.__asyncLoader,
	Jn = (t) => t.type.__isKeepAlive;
function ql(t, e) {
	Jo(t, 'a', e);
}
function Kl(t, e) {
	Jo(t, 'da', e);
}
function Jo(t, e, n = mt) {
	const i =
		t.__wdc ||
		(t.__wdc = () => {
			let s = n;
			for (; s; ) {
				if (s.isDeactivated) return;
				s = s.parent;
			}
			return t();
		});
	if ((Qn(e, i, n), n)) {
		let s = n.parent;
		for (; s && s.parent; )
			Jn(s.parent.vnode) && zl(i, e, n, s), (s = s.parent);
	}
}
function zl(t, e, n, i) {
	const s = Qn(e, t, i, !0);
	Go(() => {
		Ni(i[e], s);
	}, n);
}
function Qn(t, e, n = mt, i = !1) {
	if (n) {
		const s = n[t] || (n[t] = []),
			o =
				e.__weh ||
				(e.__weh = (...r) => {
					if (n.isUnmounted) return;
					Be(), He(n);
					const l = St(e, n, t, r);
					return Te(), Ue(), l;
				});
		return i ? s.unshift(o) : s.push(o), o;
	}
}
const Zt =
		(t) =>
		(e, n = mt) =>
			(!un || t === 'sp') && Qn(t, (...i) => e(...i), n),
	Vl = Zt('bm'),
	Qo = Zt('m'),
	Yl = Zt('bu'),
	Xl = Zt('u'),
	Zo = Zt('bum'),
	Go = Zt('um'),
	Jl = Zt('sp'),
	Ql = Zt('rtg'),
	Zl = Zt('rtc');
function Gl(t, e = mt) {
	Qn('ec', t, e);
}
function ue(t, e) {
	const n = Pt;
	if (n === null) return t;
	const i = Gn(n) || n.proxy,
		s = t.dirs || (t.dirs = []);
	for (let o = 0; o < e.length; o++) {
		let [r, l, c, f = G] = e[o];
		j(r) && (r = { mounted: r, updated: r }),
			r.deep && ye(l),
			s.push({
				dir: r,
				instance: i,
				value: l,
				oldValue: void 0,
				arg: c,
				modifiers: f,
			});
	}
	return t;
}
function fe(t, e, n, i) {
	const s = t.dirs,
		o = e && e.dirs;
	for (let r = 0; r < s.length; r++) {
		const l = s[r];
		o && (l.oldValue = o[r].value);
		let c = l.dir[i];
		c && (Be(), St(c, n, 8, [t.el, l, t, e]), Ue());
	}
}
const tc = Symbol();
function ec(t, e, n, i) {
	let s;
	const o = n && n[i];
	if (P(t) || pt(t)) {
		s = new Array(t.length);
		for (let r = 0, l = t.length; r < l; r++)
			s[r] = e(t[r], r, void 0, o && o[r]);
	} else if (typeof t == 'number') {
		s = new Array(t);
		for (let r = 0; r < t; r++) s[r] = e(r + 1, r, void 0, o && o[r]);
	} else if (tt(t))
		if (t[Symbol.iterator])
			s = Array.from(t, (r, l) => e(r, l, void 0, o && o[l]));
		else {
			const r = Object.keys(t);
			s = new Array(r.length);
			for (let l = 0, c = r.length; l < c; l++) {
				const f = r[l];
				s[l] = e(t[f], f, l, o && o[l]);
			}
		}
	else s = [];
	return n && (n[i] = s), s;
}
const yi = (t) => (t ? (ar(t) ? Gn(t) || t.proxy : yi(t.parent)) : null),
	Dn = gt(Object.create(null), {
		$: (t) => t,
		$el: (t) => t.vnode.el,
		$data: (t) => t.data,
		$props: (t) => t.props,
		$attrs: (t) => t.attrs,
		$slots: (t) => t.slots,
		$refs: (t) => t.refs,
		$parent: (t) => yi(t.parent),
		$root: (t) => yi(t.root),
		$emit: (t) => t.emit,
		$options: (t) => Ji(t),
		$forceUpdate: (t) => t.f || (t.f = () => Xi(t.update)),
		$nextTick: (t) => t.n || (t.n = Al.bind(t.proxy)),
		$watch: (t) => Bl.bind(t),
	}),
	nc = {
		get({ _: t }, e) {
			const {
				ctx: n,
				setupState: i,
				data: s,
				props: o,
				accessCache: r,
				type: l,
				appContext: c,
			} = t;
			let f;
			if (e[0] !== '$') {
				const C = r[e];
				if (C !== void 0)
					switch (C) {
						case 1:
							return i[e];
						case 2:
							return s[e];
						case 4:
							return n[e];
						case 3:
							return o[e];
					}
				else {
					if (i !== G && Y(i, e)) return (r[e] = 1), i[e];
					if (s !== G && Y(s, e)) return (r[e] = 2), s[e];
					if ((f = t.propsOptions[0]) && Y(f, e))
						return (r[e] = 3), o[e];
					if (n !== G && Y(n, e)) return (r[e] = 4), n[e];
					Ci && (r[e] = 0);
				}
			}
			const h = Dn[e];
			let m, p;
			if (h) return e === '$attrs' && Ot(t, 'get', e), h(t);
			if ((m = l.__cssModules) && (m = m[e])) return m;
			if (n !== G && Y(n, e)) return (r[e] = 4), n[e];
			if (((p = c.config.globalProperties), Y(p, e))) return p[e];
		},
		set({ _: t }, e, n) {
			const { data: i, setupState: s, ctx: o } = t;
			return s !== G && Y(s, e)
				? ((s[e] = n), !0)
				: i !== G && Y(i, e)
				? ((i[e] = n), !0)
				: Y(t.props, e) || (e[0] === '$' && e.slice(1) in t)
				? !1
				: ((o[e] = n), !0);
		},
		has(
			{
				_: {
					data: t,
					setupState: e,
					accessCache: n,
					ctx: i,
					appContext: s,
					propsOptions: o,
				},
			},
			r
		) {
			let l;
			return (
				!!n[r] ||
				(t !== G && Y(t, r)) ||
				(e !== G && Y(e, r)) ||
				((l = o[0]) && Y(l, r)) ||
				Y(i, r) ||
				Y(Dn, r) ||
				Y(s.config.globalProperties, r)
			);
		},
		defineProperty(t, e, n) {
			return (
				n.get != null
					? (t._.accessCache[e] = 0)
					: Y(n, 'value') && this.set(t, e, n.value, null),
				Reflect.defineProperty(t, e, n)
			);
		},
	};
let Ci = !0;
function ic(t) {
	const e = Ji(t),
		n = t.proxy,
		i = t.ctx;
	(Ci = !1), e.beforeCreate && Os(e.beforeCreate, t, 'bc');
	const {
		data: s,
		computed: o,
		methods: r,
		watch: l,
		provide: c,
		inject: f,
		created: h,
		beforeMount: m,
		mounted: p,
		beforeUpdate: C,
		updated: k,
		activated: O,
		deactivated: F,
		beforeDestroy: L,
		beforeUnmount: q,
		destroyed: D,
		unmounted: V,
		render: H,
		renderTracked: lt,
		renderTriggered: B,
		errorCaptured: x,
		serverPrefetch: et,
		expose: U,
		inheritAttrs: Q,
		components: T,
		directives: st,
		filters: ft,
	} = e;
	if ((f && sc(f, i, null, t.appContext.config.unwrapInjectedRef), r))
		for (const ot in r) {
			const nt = r[ot];
			j(nt) && (i[ot] = nt.bind(n));
		}
	if (s) {
		const ot = s.call(n, n);
		tt(ot) && (t.data = Ki(ot));
	}
	if (((Ci = !0), o))
		for (const ot in o) {
			const nt = o[ot],
				le = j(nt) ? nt.bind(n, n) : j(nt.get) ? nt.get.bind(n, n) : Ft,
				pn = !j(nt) && j(nt.set) ? nt.set.bind(n) : Ft,
				ce = Dc({ get: le, set: pn });
			Object.defineProperty(i, ot, {
				enumerable: !0,
				configurable: !0,
				get: () => ce.value,
				set: (Ht) => (ce.value = Ht),
			});
		}
	if (l) for (const ot in l) tr(l[ot], i, n, ot);
	if (c) {
		const ot = j(c) ? c.call(n) : c;
		Reflect.ownKeys(ot).forEach((nt) => {
			jl(nt, ot[nt]);
		});
	}
	h && Os(h, t, 'c');
	function dt(ot, nt) {
		P(nt) ? nt.forEach((le) => ot(le.bind(n))) : nt && ot(nt.bind(n));
	}
	if (
		(dt(Vl, m),
		dt(Qo, p),
		dt(Yl, C),
		dt(Xl, k),
		dt(ql, O),
		dt(Kl, F),
		dt(Gl, x),
		dt(Zl, lt),
		dt(Ql, B),
		dt(Zo, q),
		dt(Go, V),
		dt(Jl, et),
		P(U))
	)
		if (U.length) {
			const ot = t.exposed || (t.exposed = {});
			U.forEach((nt) => {
				Object.defineProperty(ot, nt, {
					get: () => n[nt],
					set: (le) => (n[nt] = le),
				});
			});
		} else t.exposed || (t.exposed = {});
	H && t.render === Ft && (t.render = H),
		Q != null && (t.inheritAttrs = Q),
		T && (t.components = T),
		st && (t.directives = st);
}
function sc(t, e, n = Ft, i = !1) {
	P(t) && (t = wi(t));
	for (const s in t) {
		const o = t[s];
		let r;
		tt(o)
			? 'default' in o
				? (r = li(o.from || s, o.default, !0))
				: (r = li(o.from || s))
			: (r = li(o)),
			rt(r) && i
				? Object.defineProperty(e, s, {
						enumerable: !0,
						configurable: !0,
						get: () => r.value,
						set: (l) => (r.value = l),
				  })
				: (e[s] = r);
	}
}
function Os(t, e, n) {
	St(P(t) ? t.map((i) => i.bind(e.proxy)) : t.bind(e.proxy), e, n);
}
function tr(t, e, n, i) {
	const s = i.includes('.') ? zo(n, i) : () => n[i];
	if (pt(t)) {
		const o = e[t];
		j(o) && ci(s, o);
	} else if (j(t)) ci(s, t.bind(n));
	else if (tt(t))
		if (P(t)) t.forEach((o) => tr(o, e, n, i));
		else {
			const o = j(t.handler) ? t.handler.bind(n) : e[t.handler];
			j(o) && ci(s, o, t);
		}
}
function Ji(t) {
	const e = t.type,
		{ mixins: n, extends: i } = e,
		{
			mixins: s,
			optionsCache: o,
			config: { optionMergeStrategies: r },
		} = t.appContext,
		l = o.get(e);
	let c;
	return (
		l
			? (c = l)
			: !s.length && !n && !i
			? (c = e)
			: ((c = {}),
			  s.length && s.forEach((f) => Fn(c, f, r, !0)),
			  Fn(c, e, r)),
		tt(e) && o.set(e, c),
		c
	);
}
function Fn(t, e, n, i = !1) {
	const { mixins: s, extends: o } = e;
	o && Fn(t, o, n, !0), s && s.forEach((r) => Fn(t, r, n, !0));
	for (const r in e)
		if (!(i && r === 'expose')) {
			const l = oc[r] || (n && n[r]);
			t[r] = l ? l(t[r], e[r]) : e[r];
		}
	return t;
}
const oc = {
	data: As,
	props: ge,
	emits: ge,
	methods: ge,
	computed: ge,
	beforeCreate: $t,
	created: $t,
	beforeMount: $t,
	mounted: $t,
	beforeUpdate: $t,
	updated: $t,
	beforeDestroy: $t,
	beforeUnmount: $t,
	destroyed: $t,
	unmounted: $t,
	activated: $t,
	deactivated: $t,
	errorCaptured: $t,
	serverPrefetch: $t,
	components: ge,
	directives: ge,
	watch: lc,
	provide: As,
	inject: rc,
};
function As(t, e) {
	return e
		? t
			? function () {
					return gt(
						j(t) ? t.call(this, this) : t,
						j(e) ? e.call(this, this) : e
					);
			  }
			: e
		: t;
}
function rc(t, e) {
	return ge(wi(t), wi(e));
}
function wi(t) {
	if (P(t)) {
		const e = {};
		for (let n = 0; n < t.length; n++) e[t[n]] = t[n];
		return e;
	}
	return t;
}
function $t(t, e) {
	return t ? [...new Set([].concat(t, e))] : e;
}
function ge(t, e) {
	return t ? gt(gt(Object.create(null), t), e) : e;
}
function lc(t, e) {
	if (!t) return e;
	if (!e) return t;
	const n = gt(Object.create(null), t);
	for (const i in e) n[i] = $t(t[i], e[i]);
	return n;
}
function cc(t, e, n, i = !1) {
	const s = {},
		o = {};
	kn(o, Zn, 1), (t.propsDefaults = Object.create(null)), er(t, e, s, o);
	for (const r in t.propsOptions[0]) r in s || (s[r] = void 0);
	n
		? (t.props = i ? s : xl(s))
		: t.type.props
		? (t.props = s)
		: (t.props = o),
		(t.attrs = o);
}
function ac(t, e, n, i) {
	const {
			props: s,
			attrs: o,
			vnode: { patchFlag: r },
		} = t,
		l = J(s),
		[c] = t.propsOptions;
	let f = !1;
	if ((i || r > 0) && !(r & 16)) {
		if (r & 8) {
			const h = t.vnode.dynamicProps;
			for (let m = 0; m < h.length; m++) {
				let p = h[m];
				if (Xn(t.emitsOptions, p)) continue;
				const C = e[p];
				if (c)
					if (Y(o, p)) C !== o[p] && ((o[p] = C), (f = !0));
					else {
						const k = De(p);
						s[k] = Ei(c, l, k, C, t, !1);
					}
				else C !== o[p] && ((o[p] = C), (f = !0));
			}
		}
	} else {
		er(t, e, s, o) && (f = !0);
		let h;
		for (const m in l)
			(!e || (!Y(e, m) && ((h = je(m)) === m || !Y(e, h)))) &&
				(c
					? n &&
					  (n[m] !== void 0 || n[h] !== void 0) &&
					  (s[m] = Ei(c, l, m, void 0, t, !0))
					: delete s[m]);
		if (o !== l)
			for (const m in o)
				(!e || (!Y(e, m) && !0)) && (delete o[m], (f = !0));
	}
	f && Qt(t, 'set', '$attrs');
}
function er(t, e, n, i) {
	const [s, o] = t.propsOptions;
	let r = !1,
		l;
	if (e)
		for (let c in e) {
			if (Tn(c)) continue;
			const f = e[c];
			let h;
			s && Y(s, (h = De(c)))
				? !o || !o.includes(h)
					? (n[h] = f)
					: ((l || (l = {}))[h] = f)
				: Xn(t.emitsOptions, c) ||
				  ((!(c in i) || f !== i[c]) && ((i[c] = f), (r = !0)));
		}
	if (o) {
		const c = J(n),
			f = l || G;
		for (let h = 0; h < o.length; h++) {
			const m = o[h];
			n[m] = Ei(s, c, m, f[m], t, !Y(f, m));
		}
	}
	return r;
}
function Ei(t, e, n, i, s, o) {
	const r = t[n];
	if (r != null) {
		const l = Y(r, 'default');
		if (l && i === void 0) {
			const c = r.default;
			if (r.type !== Function && j(c)) {
				const { propsDefaults: f } = s;
				n in f
					? (i = f[n])
					: (He(s), (i = f[n] = c.call(null, e)), Te());
			} else i = c;
		}
		r[0] &&
			(o && !l
				? (i = !1)
				: r[1] && (i === '' || i === je(n)) && (i = !0));
	}
	return i;
}
function nr(t, e, n = !1) {
	const i = e.propsCache,
		s = i.get(t);
	if (s) return s;
	const o = t.props,
		r = {},
		l = [];
	let c = !1;
	if (!j(t)) {
		const h = (m) => {
			c = !0;
			const [p, C] = nr(m, e, !0);
			gt(r, p), C && l.push(...C);
		};
		!n && e.mixins.length && e.mixins.forEach(h),
			t.extends && h(t.extends),
			t.mixins && t.mixins.forEach(h);
	}
	if (!o && !c) return tt(t) && i.set(t, Me), Me;
	if (P(o))
		for (let h = 0; h < o.length; h++) {
			const m = De(o[h]);
			Ss(m) && (r[m] = G);
		}
	else if (o)
		for (const h in o) {
			const m = De(h);
			if (Ss(m)) {
				const p = o[h],
					C = (r[m] = P(p) || j(p) ? { type: p } : p);
				if (C) {
					const k = ks(Boolean, C.type),
						O = ks(String, C.type);
					(C[0] = k > -1),
						(C[1] = O < 0 || k < O),
						(k > -1 || Y(C, 'default')) && l.push(m);
				}
			}
		}
	const f = [r, l];
	return tt(t) && i.set(t, f), f;
}
function Ss(t) {
	return t[0] !== '$';
}
function Is(t) {
	const e = t && t.toString().match(/^\s*function (\w+)/);
	return e ? e[1] : t === null ? 'null' : '';
}
function Ms(t, e) {
	return Is(t) === Is(e);
}
function ks(t, e) {
	return P(e) ? e.findIndex((n) => Ms(n, t)) : j(e) && Ms(e, t) ? 0 : -1;
}
const ir = (t) => t[0] === '_' || t === '$stable',
	Qi = (t) => (P(t) ? t.map(Bt) : [Bt(t)]),
	uc = (t, e, n) => {
		if (e._n) return e;
		const i = qo((...s) => Qi(e(...s)), n);
		return (i._c = !1), i;
	},
	sr = (t, e, n) => {
		const i = t._ctx;
		for (const s in t) {
			if (ir(s)) continue;
			const o = t[s];
			if (j(o)) e[s] = uc(s, o, i);
			else if (o != null) {
				const r = Qi(o);
				e[s] = () => r;
			}
		}
	},
	or = (t, e) => {
		const n = Qi(e);
		t.slots.default = () => n;
	},
	fc = (t, e) => {
		if (t.vnode.shapeFlag & 32) {
			const n = e._;
			n ? ((t.slots = J(e)), kn(e, '_', n)) : sr(e, (t.slots = {}));
		} else (t.slots = {}), e && or(t, e);
		kn(t.slots, Zn, 1);
	},
	dc = (t, e, n) => {
		const { vnode: i, slots: s } = t;
		let o = !0,
			r = G;
		if (i.shapeFlag & 32) {
			const l = e._;
			l
				? n && l === 1
					? (o = !1)
					: (gt(s, e), !n && l === 1 && delete s._)
				: ((o = !e.$stable), sr(e, s)),
				(r = e);
		} else e && (or(t, e), (r = { default: 1 }));
		if (o) for (const l in s) !ir(l) && !(l in r) && delete s[l];
	};
function rr() {
	return {
		app: null,
		config: {
			isNativeTag: Br,
			performance: !1,
			globalProperties: {},
			optionMergeStrategies: {},
			errorHandler: void 0,
			warnHandler: void 0,
			compilerOptions: {},
		},
		mixins: [],
		components: {},
		directives: {},
		provides: Object.create(null),
		optionsCache: new WeakMap(),
		propsCache: new WeakMap(),
		emitsCache: new WeakMap(),
	};
}
let hc = 0;
function pc(t, e) {
	return function (i, s = null) {
		j(i) || (i = Object.assign({}, i)), s != null && !tt(s) && (s = null);
		const o = rr(),
			r = new Set();
		let l = !1;
		const c = (o.app = {
			_uid: hc++,
			_component: i,
			_props: s,
			_container: null,
			_context: o,
			_instance: null,
			version: Hc,
			get config() {
				return o.config;
			},
			set config(f) {},
			use(f, ...h) {
				return (
					r.has(f) ||
						(f && j(f.install)
							? (r.add(f), f.install(c, ...h))
							: j(f) && (r.add(f), f(c, ...h))),
					c
				);
			},
			mixin(f) {
				return o.mixins.includes(f) || o.mixins.push(f), c;
			},
			component(f, h) {
				return h ? ((o.components[f] = h), c) : o.components[f];
			},
			directive(f, h) {
				return h ? ((o.directives[f] = h), c) : o.directives[f];
			},
			mount(f, h, m) {
				if (!l) {
					const p = xt(i, s);
					return (
						(p.appContext = o),
						h && e ? e(p, f) : t(p, f, m),
						(l = !0),
						(c._container = f),
						(f.__vue_app__ = c),
						Gn(p.component) || p.component.proxy
					);
				}
			},
			unmount() {
				l && (t(null, c._container), delete c._container.__vue_app__);
			},
			provide(f, h) {
				return (o.provides[f] = h), c;
			},
		});
		return c;
	};
}
function Ti(t, e, n, i, s = !1) {
	if (P(t)) {
		t.forEach((p, C) => Ti(p, e && (P(e) ? e[C] : e), n, i, s));
		return;
	}
	if (An(i) && !s) return;
	const o = i.shapeFlag & 4 ? Gn(i.component) || i.component.proxy : i.el,
		r = s ? null : o,
		{ i: l, r: c } = t,
		f = e && e.r,
		h = l.refs === G ? (l.refs = {}) : l.refs,
		m = l.setupState;
	if (
		(f != null &&
			f !== c &&
			(pt(f)
				? ((h[f] = null), Y(m, f) && (m[f] = null))
				: rt(f) && (f.value = null)),
		j(c))
	)
		se(c, l, 12, [r, h]);
	else {
		const p = pt(c),
			C = rt(c);
		if (p || C) {
			const k = () => {
				if (t.f) {
					const O = p ? h[c] : c.value;
					s
						? P(O) && Ni(O, o)
						: P(O)
						? O.includes(o) || O.push(o)
						: p
						? ((h[c] = [o]), Y(m, c) && (m[c] = h[c]))
						: ((c.value = [o]), t.k && (h[t.k] = c.value));
				} else
					p
						? ((h[c] = r), Y(m, c) && (m[c] = r))
						: C && ((c.value = r), t.k && (h[t.k] = r));
			};
			r ? ((k.id = -1), yt(k, n)) : k();
		}
	}
}
const yt = Rl;
function mc(t) {
	return gc(t);
}
function gc(t, e) {
	const n = Vr();
	n.__VUE__ = !0;
	const {
			insert: i,
			remove: s,
			patchProp: o,
			createElement: r,
			createText: l,
			createComment: c,
			setText: f,
			setElementText: h,
			parentNode: m,
			nextSibling: p,
			setScopeId: C = Ft,
			insertStaticContent: k,
		} = t,
		O = (
			u,
			d,
			g,
			v = null,
			b = null,
			y = null,
			E = !1,
			$ = null,
			w = !!d.dynamicChildren
		) => {
			if (u === d) return;
			u && !_e(u, d) && ((v = mn(u)), Ht(u, b, y, !0), (u = null)),
				d.patchFlag === -2 && ((w = !1), (d.dynamicChildren = null));
			const { type: _, ref: S, shapeFlag: A } = d;
			switch (_) {
				case Zi:
					F(u, d, g, v);
					break;
				case Xt:
					L(u, d, g, v);
					break;
				case Sn:
					u == null && q(d, g, v, E);
					break;
				case kt:
					T(u, d, g, v, b, y, E, $, w);
					break;
				default:
					A & 1
						? H(u, d, g, v, b, y, E, $, w)
						: A & 6
						? st(u, d, g, v, b, y, E, $, w)
						: (A & 64 || A & 128) &&
						  _.process(u, d, g, v, b, y, E, $, w, Oe);
			}
			S != null && b && Ti(S, u && u.ref, y, d || u, !d);
		},
		F = (u, d, g, v) => {
			if (u == null) i((d.el = l(d.children)), g, v);
			else {
				const b = (d.el = u.el);
				d.children !== u.children && f(b, d.children);
			}
		},
		L = (u, d, g, v) => {
			u == null ? i((d.el = c(d.children || '')), g, v) : (d.el = u.el);
		},
		q = (u, d, g, v) => {
			[u.el, u.anchor] = k(u.children, d, g, v, u.el, u.anchor);
		},
		D = ({ el: u, anchor: d }, g, v) => {
			let b;
			for (; u && u !== d; ) (b = p(u)), i(u, g, v), (u = b);
			i(d, g, v);
		},
		V = ({ el: u, anchor: d }) => {
			let g;
			for (; u && u !== d; ) (g = p(u)), s(u), (u = g);
			s(d);
		},
		H = (u, d, g, v, b, y, E, $, w) => {
			(E = E || d.type === 'svg'),
				u == null
					? lt(d, g, v, b, y, E, $, w)
					: et(u, d, b, y, E, $, w);
		},
		lt = (u, d, g, v, b, y, E, $) => {
			let w, _;
			const {
				type: S,
				props: A,
				shapeFlag: I,
				transition: R,
				dirs: W,
			} = u;
			if (
				((w = u.el = r(u.type, y, A && A.is, A)),
				I & 8
					? h(w, u.children)
					: I & 16 &&
					  x(
							u.children,
							w,
							null,
							v,
							b,
							y && S !== 'foreignObject',
							E,
							$
					  ),
				W && fe(u, null, v, 'created'),
				A)
			) {
				for (const Z in A)
					Z !== 'value' &&
						!Tn(Z) &&
						o(w, Z, null, A[Z], y, u.children, v, b, Kt);
				'value' in A && o(w, 'value', null, A.value),
					(_ = A.onVnodeBeforeMount) && Rt(_, v, u);
			}
			B(w, u, u.scopeId, E, v), W && fe(u, null, v, 'beforeMount');
			const it = (!b || (b && !b.pendingBranch)) && R && !R.persisted;
			it && R.beforeEnter(w),
				i(w, d, g),
				((_ = A && A.onVnodeMounted) || it || W) &&
					yt(() => {
						_ && Rt(_, v, u),
							it && R.enter(w),
							W && fe(u, null, v, 'mounted');
					}, b);
		},
		B = (u, d, g, v, b) => {
			if ((g && C(u, g), v))
				for (let y = 0; y < v.length; y++) C(u, v[y]);
			if (b) {
				let y = b.subTree;
				if (d === y) {
					const E = b.vnode;
					B(u, E, E.scopeId, E.slotScopeIds, b.parent);
				}
			}
		},
		x = (u, d, g, v, b, y, E, $, w = 0) => {
			for (let _ = w; _ < u.length; _++) {
				const S = (u[_] = $ ? ne(u[_]) : Bt(u[_]));
				O(null, S, d, g, v, b, y, E, $);
			}
		},
		et = (u, d, g, v, b, y, E) => {
			const $ = (d.el = u.el);
			let { patchFlag: w, dynamicChildren: _, dirs: S } = d;
			w |= u.patchFlag & 16;
			const A = u.props || G,
				I = d.props || G;
			let R;
			g && de(g, !1),
				(R = I.onVnodeBeforeUpdate) && Rt(R, g, d, u),
				S && fe(d, u, g, 'beforeUpdate'),
				g && de(g, !0);
			const W = b && d.type !== 'foreignObject';
			if (
				(_
					? U(u.dynamicChildren, _, $, g, v, W, y)
					: E || nt(u, d, $, null, g, v, W, y, !1),
				w > 0)
			) {
				if (w & 16) Q($, d, A, I, g, v, b);
				else if (
					(w & 2 &&
						A.class !== I.class &&
						o($, 'class', null, I.class, b),
					w & 4 && o($, 'style', A.style, I.style, b),
					w & 8)
				) {
					const it = d.dynamicProps;
					for (let Z = 0; Z < it.length; Z++) {
						const ct = it[Z],
							It = A[ct],
							Ae = I[ct];
						(Ae !== It || ct === 'value') &&
							o($, ct, It, Ae, b, u.children, g, v, Kt);
					}
				}
				w & 1 && u.children !== d.children && h($, d.children);
			} else !E && _ == null && Q($, d, A, I, g, v, b);
			((R = I.onVnodeUpdated) || S) &&
				yt(() => {
					R && Rt(R, g, d, u), S && fe(d, u, g, 'updated');
				}, v);
		},
		U = (u, d, g, v, b, y, E) => {
			for (let $ = 0; $ < d.length; $++) {
				const w = u[$],
					_ = d[$],
					S =
						w.el && (w.type === kt || !_e(w, _) || w.shapeFlag & 70)
							? m(w.el)
							: g;
				O(w, _, S, null, v, b, y, E, !0);
			}
		},
		Q = (u, d, g, v, b, y, E) => {
			if (g !== v) {
				if (g !== G)
					for (const $ in g)
						!Tn($) &&
							!($ in v) &&
							o(u, $, g[$], null, E, d.children, b, y, Kt);
				for (const $ in v) {
					if (Tn($)) continue;
					const w = v[$],
						_ = g[$];
					w !== _ &&
						$ !== 'value' &&
						o(u, $, _, w, E, d.children, b, y, Kt);
				}
				'value' in v && o(u, 'value', g.value, v.value);
			}
		},
		T = (u, d, g, v, b, y, E, $, w) => {
			const _ = (d.el = u ? u.el : l('')),
				S = (d.anchor = u ? u.anchor : l(''));
			let { patchFlag: A, dynamicChildren: I, slotScopeIds: R } = d;
			R && ($ = $ ? $.concat(R) : R),
				u == null
					? (i(_, g, v),
					  i(S, g, v),
					  x(d.children, g, S, b, y, E, $, w))
					: A > 0 && A & 64 && I && u.dynamicChildren
					? (U(u.dynamicChildren, I, g, b, y, E, $),
					  (d.key != null || (b && d === b.subTree)) && lr(u, d, !0))
					: nt(u, d, g, S, b, y, E, $, w);
		},
		st = (u, d, g, v, b, y, E, $, w) => {
			(d.slotScopeIds = $),
				u == null
					? d.shapeFlag & 512
						? b.ctx.activate(d, g, v, E, w)
						: ft(d, g, v, b, y, E, w)
					: Ke(u, d, w);
		},
		ft = (u, d, g, v, b, y, E) => {
			const $ = (u.component = Ac(u, v, b));
			if ((Jn(u) && ($.ctx.renderer = Oe), Ic($), $.asyncDep)) {
				if ((b && b.registerDep($, dt), !u.el)) {
					const w = ($.subTree = xt(Xt));
					L(null, w, d, g);
				}
				return;
			}
			dt($, u, d, g, b, y, E);
		},
		Ke = (u, d, g) => {
			const v = (d.component = u.component);
			if (Fl(u, d, g))
				if (v.asyncDep && !v.asyncResolved) {
					ot(v, d, g);
					return;
				} else (v.next = d), Il(v.update), v.update();
			else (d.el = u.el), (v.vnode = d);
		},
		dt = (u, d, g, v, b, y, E) => {
			const $ = () => {
					if (u.isMounted) {
						let { next: S, bu: A, u: I, parent: R, vnode: W } = u,
							it = S,
							Z;
						de(u, !1),
							S ? ((S.el = W.el), ot(u, S, E)) : (S = W),
							A && On(A),
							(Z = S.props && S.props.onVnodeBeforeUpdate) &&
								Rt(Z, R, S, W),
							de(u, !0);
						const ct = ri(u),
							It = u.subTree;
						(u.subTree = ct),
							O(It, ct, m(It.el), mn(It), u, b, y),
							(S.el = ct.el),
							it === null && Hl(u, ct.el),
							I && yt(I, b),
							(Z = S.props && S.props.onVnodeUpdated) &&
								yt(() => Rt(Z, R, S, W), b);
					} else {
						let S;
						const { el: A, props: I } = d,
							{ bm: R, m: W, parent: it } = u,
							Z = An(d);
						if (
							(de(u, !1),
							R && On(R),
							!Z &&
								(S = I && I.onVnodeBeforeMount) &&
								Rt(S, it, d),
							de(u, !0),
							A && si)
						) {
							const ct = () => {
								(u.subTree = ri(u)),
									si(A, u.subTree, u, b, null);
							};
							Z
								? d.type
										.__asyncLoader()
										.then(() => !u.isUnmounted && ct())
								: ct();
						} else {
							const ct = (u.subTree = ri(u));
							O(null, ct, g, v, u, b, y), (d.el = ct.el);
						}
						if (
							(W && yt(W, b), !Z && (S = I && I.onVnodeMounted))
						) {
							const ct = d;
							yt(() => Rt(S, it, ct), b);
						}
						(d.shapeFlag & 256 ||
							(it && An(it.vnode) && it.vnode.shapeFlag & 256)) &&
							u.a &&
							yt(u.a, b),
							(u.isMounted = !0),
							(d = g = v = null);
					}
				},
				w = (u.effect = new Bi($, () => Xi(_), u.scope)),
				_ = (u.update = () => w.run());
			(_.id = u.uid), de(u, !0), _();
		},
		ot = (u, d, g) => {
			d.component = u;
			const v = u.vnode.props;
			(u.vnode = d),
				(u.next = null),
				ac(u, d.props, v, g),
				dc(u, d.children, g),
				Be(),
				Cs(),
				Ue();
		},
		nt = (u, d, g, v, b, y, E, $, w = !1) => {
			const _ = u && u.children,
				S = u ? u.shapeFlag : 0,
				A = d.children,
				{ patchFlag: I, shapeFlag: R } = d;
			if (I > 0) {
				if (I & 128) {
					pn(_, A, g, v, b, y, E, $, w);
					return;
				} else if (I & 256) {
					le(_, A, g, v, b, y, E, $, w);
					return;
				}
			}
			R & 8
				? (S & 16 && Kt(_, b, y), A !== _ && h(g, A))
				: S & 16
				? R & 16
					? pn(_, A, g, v, b, y, E, $, w)
					: Kt(_, b, y, !0)
				: (S & 8 && h(g, ''), R & 16 && x(A, g, v, b, y, E, $, w));
		},
		le = (u, d, g, v, b, y, E, $, w) => {
			(u = u || Me), (d = d || Me);
			const _ = u.length,
				S = d.length,
				A = Math.min(_, S);
			let I;
			for (I = 0; I < A; I++) {
				const R = (d[I] = w ? ne(d[I]) : Bt(d[I]));
				O(u[I], R, g, null, b, y, E, $, w);
			}
			_ > S ? Kt(u, b, y, !0, !1, A) : x(d, g, v, b, y, E, $, w, A);
		},
		pn = (u, d, g, v, b, y, E, $, w) => {
			let _ = 0;
			const S = d.length;
			let A = u.length - 1,
				I = S - 1;
			for (; _ <= A && _ <= I; ) {
				const R = u[_],
					W = (d[_] = w ? ne(d[_]) : Bt(d[_]));
				if (_e(R, W)) O(R, W, g, null, b, y, E, $, w);
				else break;
				_++;
			}
			for (; _ <= A && _ <= I; ) {
				const R = u[A],
					W = (d[I] = w ? ne(d[I]) : Bt(d[I]));
				if (_e(R, W)) O(R, W, g, null, b, y, E, $, w);
				else break;
				A--, I--;
			}
			if (_ > A) {
				if (_ <= I) {
					const R = I + 1,
						W = R < S ? d[R].el : v;
					for (; _ <= I; )
						O(
							null,
							(d[_] = w ? ne(d[_]) : Bt(d[_])),
							g,
							W,
							b,
							y,
							E,
							$,
							w
						),
							_++;
				}
			} else if (_ > I) for (; _ <= A; ) Ht(u[_], b, y, !0), _++;
			else {
				const R = _,
					W = _,
					it = new Map();
				for (_ = W; _ <= I; _++) {
					const Et = (d[_] = w ? ne(d[_]) : Bt(d[_]));
					Et.key != null && it.set(Et.key, _);
				}
				let Z,
					ct = 0;
				const It = I - W + 1;
				let Ae = !1,
					fs = 0;
				const ze = new Array(It);
				for (_ = 0; _ < It; _++) ze[_] = 0;
				for (_ = R; _ <= A; _++) {
					const Et = u[_];
					if (ct >= It) {
						Ht(Et, b, y, !0);
						continue;
					}
					let Nt;
					if (Et.key != null) Nt = it.get(Et.key);
					else
						for (Z = W; Z <= I; Z++)
							if (ze[Z - W] === 0 && _e(Et, d[Z])) {
								Nt = Z;
								break;
							}
					Nt === void 0
						? Ht(Et, b, y, !0)
						: ((ze[Nt - W] = _ + 1),
						  Nt >= fs ? (fs = Nt) : (Ae = !0),
						  O(Et, d[Nt], g, null, b, y, E, $, w),
						  ct++);
				}
				const ds = Ae ? bc(ze) : Me;
				for (Z = ds.length - 1, _ = It - 1; _ >= 0; _--) {
					const Et = W + _,
						Nt = d[Et],
						hs = Et + 1 < S ? d[Et + 1].el : v;
					ze[_] === 0
						? O(null, Nt, g, hs, b, y, E, $, w)
						: Ae && (Z < 0 || _ !== ds[Z] ? ce(Nt, g, hs, 2) : Z--);
				}
			}
		},
		ce = (u, d, g, v, b = null) => {
			const {
				el: y,
				type: E,
				transition: $,
				children: w,
				shapeFlag: _,
			} = u;
			if (_ & 6) {
				ce(u.component.subTree, d, g, v);
				return;
			}
			if (_ & 128) {
				u.suspense.move(d, g, v);
				return;
			}
			if (_ & 64) {
				E.move(u, d, g, Oe);
				return;
			}
			if (E === kt) {
				i(y, d, g);
				for (let A = 0; A < w.length; A++) ce(w[A], d, g, v);
				i(u.anchor, d, g);
				return;
			}
			if (E === Sn) {
				D(u, d, g);
				return;
			}
			if (v !== 2 && _ & 1 && $)
				if (v === 0)
					$.beforeEnter(y), i(y, d, g), yt(() => $.enter(y), b);
				else {
					const { leave: A, delayLeave: I, afterLeave: R } = $,
						W = () => i(y, d, g),
						it = () => {
							A(y, () => {
								W(), R && R();
							});
						};
					I ? I(y, W, it) : it();
				}
			else i(y, d, g);
		},
		Ht = (u, d, g, v = !1, b = !1) => {
			const {
				type: y,
				props: E,
				ref: $,
				children: w,
				dynamicChildren: _,
				shapeFlag: S,
				patchFlag: A,
				dirs: I,
			} = u;
			if (($ != null && Ti($, null, g, u, !0), S & 256)) {
				d.ctx.deactivate(u);
				return;
			}
			const R = S & 1 && I,
				W = !An(u);
			let it;
			if (
				(W && (it = E && E.onVnodeBeforeUnmount) && Rt(it, d, u), S & 6)
			)
				Lr(u.component, g, v);
			else {
				if (S & 128) {
					u.suspense.unmount(g, v);
					return;
				}
				R && fe(u, null, d, 'beforeUnmount'),
					S & 64
						? u.type.remove(u, d, g, b, Oe, v)
						: _ && (y !== kt || (A > 0 && A & 64))
						? Kt(_, d, g, !1, !0)
						: ((y === kt && A & 384) || (!b && S & 16)) &&
						  Kt(w, d, g),
					v && as(u);
			}
			((W && (it = E && E.onVnodeUnmounted)) || R) &&
				yt(() => {
					it && Rt(it, d, u), R && fe(u, null, d, 'unmounted');
				}, g);
		},
		as = (u) => {
			const { type: d, el: g, anchor: v, transition: b } = u;
			if (d === kt) {
				kr(g, v);
				return;
			}
			if (d === Sn) {
				V(u);
				return;
			}
			const y = () => {
				s(g), b && !b.persisted && b.afterLeave && b.afterLeave();
			};
			if (u.shapeFlag & 1 && b && !b.persisted) {
				const { leave: E, delayLeave: $ } = b,
					w = () => E(g, y);
				$ ? $(u.el, y, w) : w();
			} else y();
		},
		kr = (u, d) => {
			let g;
			for (; u !== d; ) (g = p(u)), s(u), (u = g);
			s(d);
		},
		Lr = (u, d, g) => {
			const { bum: v, scope: b, update: y, subTree: E, um: $ } = u;
			v && On(v),
				b.stop(),
				y && ((y.active = !1), Ht(E, u, d, g)),
				$ && yt($, d),
				yt(() => {
					u.isUnmounted = !0;
				}, d),
				d &&
					d.pendingBranch &&
					!d.isUnmounted &&
					u.asyncDep &&
					!u.asyncResolved &&
					u.suspenseId === d.pendingId &&
					(d.deps--, d.deps === 0 && d.resolve());
		},
		Kt = (u, d, g, v = !1, b = !1, y = 0) => {
			for (let E = y; E < u.length; E++) Ht(u[E], d, g, v, b);
		},
		mn = (u) =>
			u.shapeFlag & 6
				? mn(u.component.subTree)
				: u.shapeFlag & 128
				? u.suspense.next()
				: p(u.anchor || u.el),
		us = (u, d, g) => {
			u == null
				? d._vnode && Ht(d._vnode, null, null, !0)
				: O(d._vnode || null, u, d, null, null, null, g),
				Cs(),
				jo(),
				(d._vnode = u);
		},
		Oe = {
			p: O,
			um: Ht,
			m: ce,
			r: as,
			mt: ft,
			mc: x,
			pc: nt,
			pbc: U,
			n: mn,
			o: t,
		};
	let ii, si;
	return (
		e && ([ii, si] = e(Oe)),
		{ render: us, hydrate: ii, createApp: pc(us, ii) }
	);
}
function de({ effect: t, update: e }, n) {
	t.allowRecurse = e.allowRecurse = n;
}
function lr(t, e, n = !1) {
	const i = t.children,
		s = e.children;
	if (P(i) && P(s))
		for (let o = 0; o < i.length; o++) {
			const r = i[o];
			let l = s[o];
			l.shapeFlag & 1 &&
				!l.dynamicChildren &&
				((l.patchFlag <= 0 || l.patchFlag === 32) &&
					((l = s[o] = ne(s[o])), (l.el = r.el)),
				n || lr(r, l));
		}
}
function bc(t) {
	const e = t.slice(),
		n = [0];
	let i, s, o, r, l;
	const c = t.length;
	for (i = 0; i < c; i++) {
		const f = t[i];
		if (f !== 0) {
			if (((s = n[n.length - 1]), t[s] < f)) {
				(e[i] = s), n.push(i);
				continue;
			}
			for (o = 0, r = n.length - 1; o < r; )
				(l = (o + r) >> 1), t[n[l]] < f ? (o = l + 1) : (r = l);
			f < t[n[o]] && (o > 0 && (e[i] = n[o - 1]), (n[o] = i));
		}
	}
	for (o = n.length, r = n[o - 1]; o-- > 0; ) (n[o] = r), (r = e[r]);
	return n;
}
const vc = (t) => t.__isTeleport,
	kt = Symbol(void 0),
	Zi = Symbol(void 0),
	Xt = Symbol(void 0),
	Sn = Symbol(void 0),
	Ze = [];
let Dt = null;
function Hn(t = !1) {
	Ze.push((Dt = t ? null : []));
}
function _c() {
	Ze.pop(), (Dt = Ze[Ze.length - 1] || null);
}
let an = 1;
function Ls(t) {
	an += t;
}
function $c(t) {
	return (
		(t.dynamicChildren = an > 0 ? Dt || Me : null),
		_c(),
		an > 0 && Dt && Dt.push(t),
		t
	);
}
function Nn(t, e, n, i, s, o) {
	return $c(K(t, e, n, i, s, o, !0));
}
function Oi(t) {
	return t ? t.__v_isVNode === !0 : !1;
}
function _e(t, e) {
	return t.type === e.type && t.key === e.key;
}
const Zn = '__vInternal',
	cr = ({ key: t }) => (t != null ? t : null),
	In = ({ ref: t, ref_key: e, ref_for: n }) =>
		t != null
			? pt(t) || rt(t) || j(t)
				? { i: Pt, r: t, k: e, f: !!n }
				: t
			: null;
function K(
	t,
	e = null,
	n = null,
	i = 0,
	s = null,
	o = t === kt ? 0 : 1,
	r = !1,
	l = !1
) {
	const c = {
		__v_isVNode: !0,
		__v_skip: !0,
		type: t,
		props: e,
		key: e && cr(e),
		ref: e && In(e),
		scopeId: Wo,
		slotScopeIds: null,
		children: n,
		component: null,
		suspense: null,
		ssContent: null,
		ssFallback: null,
		dirs: null,
		transition: null,
		el: null,
		anchor: null,
		target: null,
		targetAnchor: null,
		staticCount: 0,
		shapeFlag: o,
		patchFlag: i,
		dynamicProps: s,
		dynamicChildren: null,
		appContext: null,
	};
	return (
		l
			? (Gi(c, n), o & 128 && t.normalize(c))
			: n && (c.shapeFlag |= pt(n) ? 8 : 16),
		an > 0 &&
			!r &&
			Dt &&
			(c.patchFlag > 0 || o & 6) &&
			c.patchFlag !== 32 &&
			Dt.push(c),
		c
	);
}
const xt = xc;
function xc(t, e = null, n = null, i = 0, s = null, o = !1) {
	if (((!t || t === tc) && (t = Xt), Oi(t))) {
		const l = re(t, e, !0);
		return (
			n && Gi(l, n),
			an > 0 &&
				!o &&
				Dt &&
				(l.shapeFlag & 6 ? (Dt[Dt.indexOf(t)] = l) : Dt.push(l)),
			(l.patchFlag |= -2),
			l
		);
	}
	if ((Pc(t) && (t = t.__vccOpts), e)) {
		e = yc(e);
		let { class: l, style: c } = e;
		l && !pt(l) && (e.class = Fi(l)),
			tt(c) && (ko(c) && !P(c) && (c = gt({}, c)), (e.style = Di(c)));
	}
	const r = pt(t) ? 1 : Nl(t) ? 128 : vc(t) ? 64 : tt(t) ? 4 : j(t) ? 2 : 0;
	return K(t, e, n, i, s, r, o, !0);
}
function yc(t) {
	return t ? (ko(t) || Zn in t ? gt({}, t) : t) : null;
}
function re(t, e, n = !1) {
	const { props: i, ref: s, patchFlag: o, children: r } = t,
		l = e ? Ec(i || {}, e) : i;
	return {
		__v_isVNode: !0,
		__v_skip: !0,
		type: t.type,
		props: l,
		key: l && cr(l),
		ref:
			e && e.ref
				? n && s
					? P(s)
						? s.concat(In(e))
						: [s, In(e)]
					: In(e)
				: s,
		scopeId: t.scopeId,
		slotScopeIds: t.slotScopeIds,
		children: r,
		target: t.target,
		targetAnchor: t.targetAnchor,
		staticCount: t.staticCount,
		shapeFlag: t.shapeFlag,
		patchFlag: e && t.type !== kt ? (o === -1 ? 16 : o | 16) : o,
		dynamicProps: t.dynamicProps,
		dynamicChildren: t.dynamicChildren,
		appContext: t.appContext,
		dirs: t.dirs,
		transition: t.transition,
		component: t.component,
		suspense: t.suspense,
		ssContent: t.ssContent && re(t.ssContent),
		ssFallback: t.ssFallback && re(t.ssFallback),
		el: t.el,
		anchor: t.anchor,
	};
}
function Cc(t = ' ', e = 0) {
	return xt(Zi, null, t, e);
}
function wc(t, e) {
	const n = xt(Sn, null, t);
	return (n.staticCount = e), n;
}
function Bt(t) {
	return t == null || typeof t == 'boolean'
		? xt(Xt)
		: P(t)
		? xt(kt, null, t.slice())
		: typeof t == 'object'
		? ne(t)
		: xt(Zi, null, String(t));
}
function ne(t) {
	return (t.el === null && t.patchFlag !== -1) || t.memo ? t : re(t);
}
function Gi(t, e) {
	let n = 0;
	const { shapeFlag: i } = t;
	if (e == null) e = null;
	else if (P(e)) n = 16;
	else if (typeof e == 'object')
		if (i & 65) {
			const s = e.default;
			s && (s._c && (s._d = !1), Gi(t, s()), s._c && (s._d = !0));
			return;
		} else {
			n = 32;
			const s = e._;
			!s && !(Zn in e)
				? (e._ctx = Pt)
				: s === 3 &&
				  Pt &&
				  (Pt.slots._ === 1
						? (e._ = 1)
						: ((e._ = 2), (t.patchFlag |= 1024)));
		}
	else
		j(e)
			? ((e = { default: e, _ctx: Pt }), (n = 32))
			: ((e = String(e)), i & 64 ? ((n = 16), (e = [Cc(e)])) : (n = 8));
	(t.children = e), (t.shapeFlag |= n);
}
function Ec(...t) {
	const e = {};
	for (let n = 0; n < t.length; n++) {
		const i = t[n];
		for (const s in i)
			if (s === 'class')
				e.class !== i.class && (e.class = Fi([e.class, i.class]));
			else if (s === 'style') e.style = Di([e.style, i.style]);
			else if (qn(s)) {
				const o = e[s],
					r = i[s];
				r &&
					o !== r &&
					!(P(o) && o.includes(r)) &&
					(e[s] = o ? [].concat(o, r) : r);
			} else s !== '' && (e[s] = i[s]);
	}
	return e;
}
function Rt(t, e, n, i = null) {
	St(t, e, 7, [n, i]);
}
const Tc = rr();
let Oc = 0;
function Ac(t, e, n) {
	const i = t.type,
		s = (e ? e.appContext : t.appContext) || Tc,
		o = {
			uid: Oc++,
			vnode: t,
			type: i,
			parent: e,
			appContext: s,
			root: null,
			next: null,
			subTree: null,
			effect: null,
			update: null,
			scope: new Yr(!0),
			render: null,
			proxy: null,
			exposed: null,
			exposeProxy: null,
			withProxy: null,
			provides: e ? e.provides : Object.create(s.provides),
			accessCache: null,
			renderCache: [],
			components: null,
			directives: null,
			propsOptions: nr(i, s),
			emitsOptions: Uo(i, s),
			emit: null,
			emitted: null,
			propsDefaults: G,
			inheritAttrs: i.inheritAttrs,
			ctx: G,
			data: G,
			props: G,
			attrs: G,
			slots: G,
			refs: G,
			setupState: G,
			setupContext: null,
			suspense: n,
			suspenseId: n ? n.pendingId : 0,
			asyncDep: null,
			asyncResolved: !1,
			isMounted: !1,
			isUnmounted: !1,
			isDeactivated: !1,
			bc: null,
			c: null,
			bm: null,
			m: null,
			bu: null,
			u: null,
			um: null,
			bum: null,
			da: null,
			a: null,
			rtg: null,
			rtc: null,
			ec: null,
			sp: null,
		};
	return (
		(o.ctx = { _: o }),
		(o.root = e ? e.root : o),
		(o.emit = Ll.bind(null, o)),
		t.ce && t.ce(o),
		o
	);
}
let mt = null;
const Sc = () => mt || Pt,
	He = (t) => {
		(mt = t), t.scope.on();
	},
	Te = () => {
		mt && mt.scope.off(), (mt = null);
	};
function ar(t) {
	return t.vnode.shapeFlag & 4;
}
let un = !1;
function Ic(t, e = !1) {
	un = e;
	const { props: n, children: i } = t.vnode,
		s = ar(t);
	cc(t, n, s, e), fc(t, i);
	const o = s ? Mc(t, e) : void 0;
	return (un = !1), o;
}
function Mc(t, e) {
	const n = t.type;
	(t.accessCache = Object.create(null)), (t.proxy = Lo(new Proxy(t.ctx, nc)));
	const { setup: i } = n;
	if (i) {
		const s = (t.setupContext = i.length > 1 ? Lc(t) : null);
		He(t), Be();
		const o = se(i, t, 0, [t.props, s]);
		if ((Ue(), Te(), bo(o))) {
			if ((o.then(Te, Te), e))
				return o
					.then((r) => {
						Ps(t, r, e);
					})
					.catch((r) => {
						Yn(r, t, 0);
					});
			t.asyncDep = o;
		} else Ps(t, o, e);
	} else ur(t, e);
}
function Ps(t, e, n) {
	j(e)
		? t.type.__ssrInlineRender
			? (t.ssrRender = e)
			: (t.render = e)
		: tt(e) && (t.setupState = Fo(e)),
		ur(t, n);
}
let Ds;
function ur(t, e, n) {
	const i = t.type;
	if (!t.render) {
		if (!e && Ds && !i.render) {
			const s = i.template || Ji(t).template;
			if (s) {
				const { isCustomElement: o, compilerOptions: r } =
						t.appContext.config,
					{ delimiters: l, compilerOptions: c } = i,
					f = gt(gt({ isCustomElement: o, delimiters: l }, r), c);
				i.render = Ds(s, f);
			}
		}
		t.render = i.render || Ft;
	}
	He(t), Be(), ic(t), Ue(), Te();
}
function kc(t) {
	return new Proxy(t.attrs, {
		get(e, n) {
			return Ot(t, 'get', '$attrs'), e[n];
		},
	});
}
function Lc(t) {
	const e = (i) => {
		t.exposed = i || {};
	};
	let n;
	return {
		get attrs() {
			return n || (n = kc(t));
		},
		slots: t.slots,
		emit: t.emit,
		expose: e,
	};
}
function Gn(t) {
	if (t.exposed)
		return (
			t.exposeProxy ||
			(t.exposeProxy = new Proxy(Fo(Lo(t.exposed)), {
				get(e, n) {
					if (n in e) return e[n];
					if (n in Dn) return Dn[n](t);
				},
			}))
		);
}
function Pc(t) {
	return j(t) && '__vccOpts' in t;
}
const Dc = (t, e) => Tl(t, e, un);
function Fc(t, e, n) {
	const i = arguments.length;
	return i === 2
		? tt(e) && !P(e)
			? Oi(e)
				? xt(t, null, [e])
				: xt(t, e)
			: xt(t, null, e)
		: (i > 3
				? (n = Array.prototype.slice.call(arguments, 2))
				: i === 3 && Oi(n) && (n = [n]),
		  xt(t, e, n));
}
const Hc = '3.2.40',
	Nc = 'http://www.w3.org/2000/svg',
	$e = typeof document < 'u' ? document : null,
	Fs = $e && $e.createElement('template'),
	Rc = {
		insert: (t, e, n) => {
			e.insertBefore(t, n || null);
		},
		remove: (t) => {
			const e = t.parentNode;
			e && e.removeChild(t);
		},
		createElement: (t, e, n, i) => {
			const s = e
				? $e.createElementNS(Nc, t)
				: $e.createElement(t, n ? { is: n } : void 0);
			return (
				t === 'select' &&
					i &&
					i.multiple != null &&
					s.setAttribute('multiple', i.multiple),
				s
			);
		},
		createText: (t) => $e.createTextNode(t),
		createComment: (t) => $e.createComment(t),
		setText: (t, e) => {
			t.nodeValue = e;
		},
		setElementText: (t, e) => {
			t.textContent = e;
		},
		parentNode: (t) => t.parentNode,
		nextSibling: (t) => t.nextSibling,
		querySelector: (t) => $e.querySelector(t),
		setScopeId(t, e) {
			t.setAttribute(e, '');
		},
		insertStaticContent(t, e, n, i, s, o) {
			const r = n ? n.previousSibling : e.lastChild;
			if (s && (s === o || s.nextSibling))
				for (
					;
					e.insertBefore(s.cloneNode(!0), n),
						!(s === o || !(s = s.nextSibling));

				);
			else {
				Fs.innerHTML = i ? `<svg>${t}</svg>` : t;
				const l = Fs.content;
				if (i) {
					const c = l.firstChild;
					for (; c.firstChild; ) l.appendChild(c.firstChild);
					l.removeChild(c);
				}
				e.insertBefore(l, n);
			}
			return [
				r ? r.nextSibling : e.firstChild,
				n ? n.previousSibling : e.lastChild,
			];
		},
	};
function jc(t, e, n) {
	const i = t._vtc;
	i && (e = (e ? [e, ...i] : [...i]).join(' ')),
		e == null
			? t.removeAttribute('class')
			: n
			? t.setAttribute('class', e)
			: (t.className = e);
}
function Bc(t, e, n) {
	const i = t.style,
		s = pt(n);
	if (n && !s) {
		for (const o in n) Ai(i, o, n[o]);
		if (e && !pt(e)) for (const o in e) n[o] == null && Ai(i, o, '');
	} else {
		const o = i.display;
		s ? e !== n && (i.cssText = n) : e && t.removeAttribute('style'),
			'_vod' in t && (i.display = o);
	}
}
const Hs = /\s*!important$/;
function Ai(t, e, n) {
	if (P(n)) n.forEach((i) => Ai(t, e, i));
	else if ((n == null && (n = ''), e.startsWith('--'))) t.setProperty(e, n);
	else {
		const i = Uc(t, e);
		Hs.test(n)
			? t.setProperty(je(i), n.replace(Hs, ''), 'important')
			: (t[i] = n);
	}
}
const Ns = ['Webkit', 'Moz', 'ms'],
	ui = {};
function Uc(t, e) {
	const n = ui[e];
	if (n) return n;
	let i = De(e);
	if (i !== 'filter' && i in t) return (ui[e] = i);
	i = $o(i);
	for (let s = 0; s < Ns.length; s++) {
		const o = Ns[s] + i;
		if (o in t) return (ui[e] = o);
	}
	return e;
}
const Rs = 'http://www.w3.org/1999/xlink';
function Wc(t, e, n, i, s) {
	if (i && e.startsWith('xlink:'))
		n == null
			? t.removeAttributeNS(Rs, e.slice(6, e.length))
			: t.setAttributeNS(Rs, e, n);
	else {
		const o = Dr(e);
		n == null || (o && !mo(n))
			? t.removeAttribute(e)
			: t.setAttribute(e, o ? '' : n);
	}
}
function qc(t, e, n, i, s, o, r) {
	if (e === 'innerHTML' || e === 'textContent') {
		i && r(i, s, o), (t[e] = n == null ? '' : n);
		return;
	}
	if (e === 'value' && t.tagName !== 'PROGRESS' && !t.tagName.includes('-')) {
		t._value = n;
		const c = n == null ? '' : n;
		(t.value !== c || t.tagName === 'OPTION') && (t.value = c),
			n == null && t.removeAttribute(e);
		return;
	}
	let l = !1;
	if (n === '' || n == null) {
		const c = typeof t[e];
		c === 'boolean'
			? (n = mo(n))
			: n == null && c === 'string'
			? ((n = ''), (l = !0))
			: c === 'number' && ((n = 0), (l = !0));
	}
	try {
		t[e] = n;
	} catch {}
	l && t.removeAttribute(e);
}
const [fr, Kc] = (() => {
	let t = Date.now,
		e = !1;
	if (typeof window < 'u') {
		Date.now() > document.createEvent('Event').timeStamp &&
			(t = performance.now.bind(performance));
		const n = navigator.userAgent.match(/firefox\/(\d+)/i);
		e = !!(n && Number(n[1]) <= 53);
	}
	return [t, e];
})();
let Si = 0;
const zc = Promise.resolve(),
	Vc = () => {
		Si = 0;
	},
	Yc = () => Si || (zc.then(Vc), (Si = fr()));
function xe(t, e, n, i) {
	t.addEventListener(e, n, i);
}
function Xc(t, e, n, i) {
	t.removeEventListener(e, n, i);
}
function Jc(t, e, n, i, s = null) {
	const o = t._vei || (t._vei = {}),
		r = o[e];
	if (i && r) r.value = i;
	else {
		const [l, c] = Qc(e);
		if (i) {
			const f = (o[e] = Zc(i, s));
			xe(t, l, f, c);
		} else r && (Xc(t, l, r, c), (o[e] = void 0));
	}
}
const js = /(?:Once|Passive|Capture)$/;
function Qc(t) {
	let e;
	if (js.test(t)) {
		e = {};
		let i;
		for (; (i = t.match(js)); )
			(t = t.slice(0, t.length - i[0].length)),
				(e[i[0].toLowerCase()] = !0);
	}
	return [t[2] === ':' ? t.slice(3) : je(t.slice(2)), e];
}
function Zc(t, e) {
	const n = (i) => {
		const s = i.timeStamp || fr();
		(Kc || s >= n.attached - 1) && St(Gc(i, n.value), e, 5, [i]);
	};
	return (n.value = t), (n.attached = Yc()), n;
}
function Gc(t, e) {
	if (P(e)) {
		const n = t.stopImmediatePropagation;
		return (
			(t.stopImmediatePropagation = () => {
				n.call(t), (t._stopped = !0);
			}),
			e.map((i) => (s) => !s._stopped && i && i(s))
		);
	} else return e;
}
const Bs = /^on[a-z]/,
	ta = (t, e, n, i, s = !1, o, r, l, c) => {
		e === 'class'
			? jc(t, i, s)
			: e === 'style'
			? Bc(t, n, i)
			: qn(e)
			? Hi(e) || Jc(t, e, n, i, r)
			: (
					e[0] === '.'
						? ((e = e.slice(1)), !0)
						: e[0] === '^'
						? ((e = e.slice(1)), !1)
						: ea(t, e, i, s)
			  )
			? qc(t, e, i, o, r, l, c)
			: (e === 'true-value'
					? (t._trueValue = i)
					: e === 'false-value' && (t._falseValue = i),
			  Wc(t, e, i, s));
	};
function ea(t, e, n, i) {
	return i
		? !!(
				e === 'innerHTML' ||
				e === 'textContent' ||
				(e in t && Bs.test(e) && j(n))
		  )
		: e === 'spellcheck' ||
		  e === 'draggable' ||
		  e === 'translate' ||
		  e === 'form' ||
		  (e === 'list' && t.tagName === 'INPUT') ||
		  (e === 'type' && t.tagName === 'TEXTAREA') ||
		  (Bs.test(e) && pt(n))
		? !1
		: e in t;
}
const te = 'transition',
	Ve = 'animation',
	ts = (t, { slots: e }) => Fc(Vo, na(t), e);
ts.displayName = 'Transition';
const dr = {
	name: String,
	type: String,
	css: { type: Boolean, default: !0 },
	duration: [String, Number, Object],
	enterFromClass: String,
	enterActiveClass: String,
	enterToClass: String,
	appearFromClass: String,
	appearActiveClass: String,
	appearToClass: String,
	leaveFromClass: String,
	leaveActiveClass: String,
	leaveToClass: String,
};
ts.props = gt({}, Vo.props, dr);
const he = (t, e = []) => {
		P(t) ? t.forEach((n) => n(...e)) : t && t(...e);
	},
	Us = (t) => (t ? (P(t) ? t.some((e) => e.length > 1) : t.length > 1) : !1);
function na(t) {
	const e = {};
	for (const T in t) T in dr || (e[T] = t[T]);
	if (t.css === !1) return e;
	const {
			name: n = 'v',
			type: i,
			duration: s,
			enterFromClass: o = `${n}-enter-from`,
			enterActiveClass: r = `${n}-enter-active`,
			enterToClass: l = `${n}-enter-to`,
			appearFromClass: c = o,
			appearActiveClass: f = r,
			appearToClass: h = l,
			leaveFromClass: m = `${n}-leave-from`,
			leaveActiveClass: p = `${n}-leave-active`,
			leaveToClass: C = `${n}-leave-to`,
		} = t,
		k = ia(s),
		O = k && k[0],
		F = k && k[1],
		{
			onBeforeEnter: L,
			onEnter: q,
			onEnterCancelled: D,
			onLeave: V,
			onLeaveCancelled: H,
			onBeforeAppear: lt = L,
			onAppear: B = q,
			onAppearCancelled: x = D,
		} = e,
		et = (T, st, ft) => {
			pe(T, st ? h : l), pe(T, st ? f : r), ft && ft();
		},
		U = (T, st) => {
			(T._isLeaving = !1), pe(T, m), pe(T, C), pe(T, p), st && st();
		},
		Q = (T) => (st, ft) => {
			const Ke = T ? B : q,
				dt = () => et(st, T, ft);
			he(Ke, [st, dt]),
				Ws(() => {
					pe(st, T ? c : o),
						ee(st, T ? h : l),
						Us(Ke) || qs(st, i, O, dt);
				});
		};
	return gt(e, {
		onBeforeEnter(T) {
			he(L, [T]), ee(T, o), ee(T, r);
		},
		onBeforeAppear(T) {
			he(lt, [T]), ee(T, c), ee(T, f);
		},
		onEnter: Q(!1),
		onAppear: Q(!0),
		onLeave(T, st) {
			T._isLeaving = !0;
			const ft = () => U(T, st);
			ee(T, m),
				ra(),
				ee(T, p),
				Ws(() => {
					!T._isLeaving ||
						(pe(T, m), ee(T, C), Us(V) || qs(T, i, F, ft));
				}),
				he(V, [T, ft]);
		},
		onEnterCancelled(T) {
			et(T, !1), he(D, [T]);
		},
		onAppearCancelled(T) {
			et(T, !0), he(x, [T]);
		},
		onLeaveCancelled(T) {
			U(T), he(H, [T]);
		},
	});
}
function ia(t) {
	if (t == null) return null;
	if (tt(t)) return [fi(t.enter), fi(t.leave)];
	{
		const e = fi(t);
		return [e, e];
	}
}
function fi(t) {
	return on(t);
}
function ee(t, e) {
	e.split(/\s+/).forEach((n) => n && t.classList.add(n)),
		(t._vtc || (t._vtc = new Set())).add(e);
}
function pe(t, e) {
	e.split(/\s+/).forEach((i) => i && t.classList.remove(i));
	const { _vtc: n } = t;
	n && (n.delete(e), n.size || (t._vtc = void 0));
}
function Ws(t) {
	requestAnimationFrame(() => {
		requestAnimationFrame(t);
	});
}
let sa = 0;
function qs(t, e, n, i) {
	const s = (t._endId = ++sa),
		o = () => {
			s === t._endId && i();
		};
	if (n) return setTimeout(o, n);
	const { type: r, timeout: l, propCount: c } = oa(t, e);
	if (!r) return i();
	const f = r + 'end';
	let h = 0;
	const m = () => {
			t.removeEventListener(f, p), o();
		},
		p = (C) => {
			C.target === t && ++h >= c && m();
		};
	setTimeout(() => {
		h < c && m();
	}, l + 1),
		t.addEventListener(f, p);
}
function oa(t, e) {
	const n = window.getComputedStyle(t),
		i = (k) => (n[k] || '').split(', '),
		s = i(te + 'Delay'),
		o = i(te + 'Duration'),
		r = Ks(s, o),
		l = i(Ve + 'Delay'),
		c = i(Ve + 'Duration'),
		f = Ks(l, c);
	let h = null,
		m = 0,
		p = 0;
	e === te
		? r > 0 && ((h = te), (m = r), (p = o.length))
		: e === Ve
		? f > 0 && ((h = Ve), (m = f), (p = c.length))
		: ((m = Math.max(r, f)),
		  (h = m > 0 ? (r > f ? te : Ve) : null),
		  (p = h ? (h === te ? o.length : c.length) : 0));
	const C = h === te && /\b(transform|all)(,|$)/.test(n[te + 'Property']);
	return { type: h, timeout: m, propCount: p, hasTransform: C };
}
function Ks(t, e) {
	for (; t.length < e.length; ) t = t.concat(t);
	return Math.max(...e.map((n, i) => zs(n) + zs(t[i])));
}
function zs(t) {
	return Number(t.slice(0, -1).replace(',', '.')) * 1e3;
}
function ra() {
	return document.body.offsetHeight;
}
const Rn = (t) => {
	const e = t.props['onUpdate:modelValue'] || !1;
	return P(e) ? (n) => On(e, n) : e;
};
function la(t) {
	t.target.composing = !0;
}
function Vs(t) {
	const e = t.target;
	e.composing && ((e.composing = !1), e.dispatchEvent(new Event('input')));
}
const xn = {
		created(t, { modifiers: { lazy: e, trim: n, number: i } }, s) {
			t._assign = Rn(s);
			const o = i || (s.props && s.props.type === 'number');
			xe(t, e ? 'change' : 'input', (r) => {
				if (r.target.composing) return;
				let l = t.value;
				n && (l = l.trim()), o && (l = on(l)), t._assign(l);
			}),
				n &&
					xe(t, 'change', () => {
						t.value = t.value.trim();
					}),
				e ||
					(xe(t, 'compositionstart', la),
					xe(t, 'compositionend', Vs),
					xe(t, 'change', Vs));
		},
		mounted(t, { value: e }) {
			t.value = e == null ? '' : e;
		},
		beforeUpdate(
			t,
			{ value: e, modifiers: { lazy: n, trim: i, number: s } },
			o
		) {
			if (
				((t._assign = Rn(o)),
				t.composing ||
					(document.activeElement === t &&
						t.type !== 'range' &&
						(n ||
							(i && t.value.trim() === e) ||
							((s || t.type === 'number') && on(t.value) === e))))
			)
				return;
			const r = e == null ? '' : e;
			t.value !== r && (t.value = r);
		},
	},
	ca = {
		deep: !0,
		created(t, { value: e, modifiers: { number: n } }, i) {
			const s = Kn(e);
			xe(t, 'change', () => {
				const o = Array.prototype.filter
					.call(t.options, (r) => r.selected)
					.map((r) => (n ? on(jn(r)) : jn(r)));
				t._assign(t.multiple ? (s ? new Set(o) : o) : o[0]);
			}),
				(t._assign = Rn(i));
		},
		mounted(t, { value: e }) {
			Ys(t, e);
		},
		beforeUpdate(t, e, n) {
			t._assign = Rn(n);
		},
		updated(t, { value: e }) {
			Ys(t, e);
		},
	};
function Ys(t, e) {
	const n = t.multiple;
	if (!(n && !P(e) && !Kn(e))) {
		for (let i = 0, s = t.options.length; i < s; i++) {
			const o = t.options[i],
				r = jn(o);
			if (n)
				P(e) ? (o.selected = jr(e, r) > -1) : (o.selected = e.has(r));
			else if (Wn(jn(o), e)) {
				t.selectedIndex !== i && (t.selectedIndex = i);
				return;
			}
		}
		!n && t.selectedIndex !== -1 && (t.selectedIndex = -1);
	}
}
function jn(t) {
	return '_value' in t ? t._value : t.value;
}
const Xs = {
	beforeMount(t, { value: e }, { transition: n }) {
		(t._vod = t.style.display === 'none' ? '' : t.style.display),
			n && e ? n.beforeEnter(t) : Ye(t, e);
	},
	mounted(t, { value: e }, { transition: n }) {
		n && e && n.enter(t);
	},
	updated(t, { value: e, oldValue: n }, { transition: i }) {
		!e != !n &&
			(i
				? e
					? (i.beforeEnter(t), Ye(t, !0), i.enter(t))
					: i.leave(t, () => {
							Ye(t, !1);
					  })
				: Ye(t, e));
	},
	beforeUnmount(t, { value: e }) {
		Ye(t, e);
	},
};
function Ye(t, e) {
	t.style.display = e ? t._vod : 'none';
}
const aa = gt({ patchProp: ta }, Rc);
let Js;
function ua() {
	return Js || (Js = mc(aa));
}
const fa = (...t) => {
	const e = ua().createApp(...t),
		{ mount: n } = e;
	return (
		(e.mount = (i) => {
			const s = da(i);
			if (!s) return;
			const o = e._component;
			!j(o) && !o.render && !o.template && (o.template = s.innerHTML),
				(s.innerHTML = '');
			const r = n(s, !1, s instanceof SVGElement);
			return (
				s instanceof Element &&
					(s.removeAttribute('v-cloak'),
					s.setAttribute('data-v-app', '')),
				r
			);
		}),
		e
	);
};
function da(t) {
	return pt(t) ? document.querySelector(t) : t;
}
/*!
 * mdui 1.0.2 (https://mdui.org)
 * Copyright 2016-2021 zdhxiong
 * Licensed under MIT
 */ function _t(t) {
	return typeof t == 'function';
}
function wt(t) {
	return typeof t == 'string';
}
function We(t) {
	return typeof t == 'number';
}
function ha(t) {
	return typeof t == 'boolean';
}
function z(t) {
	return typeof t > 'u';
}
function Ii(t) {
	return t === null;
}
function es(t) {
	return t instanceof Window;
}
function ns(t) {
	return t instanceof Document;
}
function qe(t) {
	return t instanceof Element;
}
function pa(t) {
	return t instanceof Node;
}
function ma() {
	return !!window.document.documentMode;
}
function hr(t) {
	return _t(t) || es(t) ? !1 : We(t.length);
}
function Jt(t) {
	return typeof t == 'object' && t !== null;
}
function Bn(t) {
	return ns(t) ? t.documentElement : t;
}
function hn(t) {
	return t
		.replace(/^-ms-/, 'ms-')
		.replace(/-([a-z])/g, (e, n) => n.toUpperCase());
}
function pr(t) {
	return t.replace(/[A-Z]/g, (e) => '-' + e.toLowerCase());
}
function ti(t, e) {
	return window.getComputedStyle(t).getPropertyValue(pr(e));
}
function mr(t) {
	return ti(t, 'box-sizing') === 'border-box';
}
function Mi(t, e, n) {
	const i = e === 'width' ? ['Left', 'Right'] : ['Top', 'Bottom'];
	return [0, 1].reduce((s, o, r) => {
		let l = n + i[r];
		return (
			n === 'border' && (l += 'Width'), s + parseFloat(ti(t, l) || '0')
		);
	}, 0);
}
function ei(t, e) {
	if (e === 'width' || e === 'height') {
		const n = t.getBoundingClientRect()[e];
		return mr(t)
			? `${n}px`
			: `${n - Mi(t, e, 'border') - Mi(t, e, 'padding')}px`;
	}
	return ti(t, e);
}
function gr(t, e) {
	const n = document.createElement(e);
	return (n.innerHTML = t), [].slice.call(n.childNodes);
}
function br() {
	return !1;
}
const ga = [
	'animationIterationCount',
	'columnCount',
	'fillOpacity',
	'flexGrow',
	'flexShrink',
	'fontWeight',
	'gridArea',
	'gridColumn',
	'gridColumnEnd',
	'gridColumnStart',
	'gridRow',
	'gridRowEnd',
	'gridRowStart',
	'lineHeight',
	'opacity',
	'order',
	'orphans',
	'widows',
	'zIndex',
	'zoom',
];
function N(t, e) {
	if (hr(t)) {
		for (let n = 0; n < t.length; n += 1)
			if (e.call(t[n], n, t[n]) === !1) return t;
	} else {
		const n = Object.keys(t);
		for (let i = 0; i < n.length; i += 1)
			if (e.call(t[n[i]], n[i], t[n[i]]) === !1) return t;
	}
	return t;
}
class ht {
	constructor(e) {
		return (
			(this.length = 0),
			e
				? (N(e, (n, i) => {
						this[n] = i;
				  }),
				  (this.length = e.length),
				  this)
				: this
		);
	}
}
function ba() {
	const t = function (e) {
		if (!e) return new ht();
		if (e instanceof ht) return e;
		if (_t(e))
			return (
				/complete|loaded|interactive/.test(document.readyState) &&
				document.body
					? e.call(document, t)
					: document.addEventListener(
							'DOMContentLoaded',
							() => e.call(document, t),
							!1
					  ),
				new ht([document])
			);
		if (wt(e)) {
			const n = e.trim();
			if (n[0] === '<' && n[n.length - 1] === '>') {
				let o = 'div';
				return (
					N(
						{
							li: 'ul',
							tr: 'tbody',
							td: 'tr',
							th: 'tr',
							tbody: 'table',
							option: 'select',
						},
						(l, c) => {
							if (n.indexOf(`<${l}`) === 0) return (o = c), !1;
						}
					),
					new ht(gr(n, o))
				);
			}
			if (!(e[0] === '#' && !e.match(/[ .<>:~]/)))
				return new ht(document.querySelectorAll(e));
			const s = document.getElementById(e.slice(1));
			return s ? new ht([s]) : new ht();
		}
		return hr(e) && !pa(e) ? new ht(e) : new ht([e]);
	};
	return (t.fn = ht.prototype), t;
}
const a = ba();
setTimeout(() => a('body').addClass('mdui-loaded'));
const M = { $: a };
a.fn.each = function (t) {
	return N(this, t);
};
function Yt(t, e) {
	return t !== e && Bn(t).contains(e);
}
function is(t, e) {
	return (
		N(e, (n, i) => {
			t.push(i);
		}),
		t
	);
}
a.fn.get = function (t) {
	return t === void 0
		? [].slice.call(this)
		: this[t >= 0 ? t : t + this.length];
};
a.fn.find = function (t) {
	const e = [];
	return (
		this.each((n, i) => {
			is(e, a(i.querySelectorAll(t)).get());
		}),
		new ht(e)
	);
};
const Ie = {};
let va = 1;
function Ge(t) {
	const e = '_mduiEventId';
	return t[e] || (t[e] = ++va), t[e];
}
function ss(t) {
	const e = t.split('.');
	return { type: e[0], ns: e.slice(1).sort().join(' ') };
}
function vr(t) {
	return new RegExp('(?:^| )' + t.replace(' ', ' .* ?') + '(?: |$)');
}
function _a(t, e, n, i) {
	const s = ss(e);
	return (Ie[Ge(t)] || []).filter(
		(o) =>
			o &&
			(!s.type || o.type === s.type) &&
			(!s.ns || vr(s.ns).test(o.ns)) &&
			(!n || Ge(o.func) === Ge(n)) &&
			(!i || o.selector === i)
	);
}
function $a(t, e, n, i, s) {
	const o = Ge(t);
	Ie[o] || (Ie[o] = []);
	let r = !1;
	Jt(i) && i.useCapture && (r = !0),
		e.split(' ').forEach((l) => {
			if (!l) return;
			const c = ss(l);
			function f(p, C) {
				n.apply(
					C,
					p._detail === void 0 ? [p] : [p].concat(p._detail)
				) === !1 && (p.preventDefault(), p.stopPropagation());
			}
			function h(p) {
				(p._ns && !vr(p._ns).test(c.ns)) ||
					((p._data = i),
					s
						? a(t)
								.find(s)
								.get()
								.reverse()
								.forEach((C) => {
									(C === p.target || Yt(C, p.target)) &&
										f(p, C);
								})
						: f(p, t));
			}
			const m = {
				type: c.type,
				ns: c.ns,
				func: n,
				selector: s,
				id: Ie[o].length,
				proxy: h,
			};
			Ie[o].push(m), t.addEventListener(m.type, h, r);
		});
}
function xa(t, e, n, i) {
	const s = Ie[Ge(t)] || [],
		o = (r) => {
			delete s[r.id], t.removeEventListener(r.type, r.proxy, !1);
		};
	e
		? e.split(' ').forEach((r) => {
				r && _a(t, r, n, i).forEach((l) => o(l));
		  })
		: s.forEach((r) => o(r));
}
a.fn.trigger = function (t, e) {
	const n = ss(t);
	let i;
	const s = { bubbles: !0, cancelable: !0 };
	return (
		['click', 'mousedown', 'mouseup', 'mousemove'].indexOf(n.type) > -1
			? (i = new MouseEvent(n.type, s))
			: ((s.detail = e), (i = new CustomEvent(n.type, s))),
		(i._detail = e),
		(i._ns = n.ns),
		this.each(function () {
			this.dispatchEvent(i);
		})
	);
};
function X(t, e, ...n) {
	return (
		n.unshift(e),
		N(n, (i, s) => {
			N(s, (o, r) => {
				z(r) || (t[o] = r);
			});
		}),
		t
	);
}
function os(t) {
	if (!Jt(t) && !Array.isArray(t)) return '';
	const e = [];
	function n(i, s) {
		let o;
		Jt(s)
			? N(s, (r, l) => {
					Array.isArray(s) && !Jt(l) ? (o = '') : (o = r),
						n(`${i}[${o}]`, l);
			  })
			: (s == null || s === ''
					? (o = '=')
					: (o = `=${encodeURIComponent(s)}`),
			  e.push(encodeURIComponent(i) + o));
	}
	return (
		Array.isArray(t)
			? N(t, function () {
					n(this.name, this.value);
			  })
			: N(t, n),
		e.join('&')
	);
}
const tn = {},
	Mt = {
		ajaxStart: 'start.mdui.ajax',
		ajaxSuccess: 'success.mdui.ajax',
		ajaxError: 'error.mdui.ajax',
		ajaxComplete: 'complete.mdui.ajax',
	};
function yn(t) {
	return ['GET', 'HEAD'].indexOf(t) >= 0;
}
function Qs(t, e) {
	return `${t}&${e}`.replace(/[&?]{1,2}/, '?');
}
function ya(t) {
	const e = {
		url: '',
		method: 'GET',
		data: '',
		processData: !0,
		async: !0,
		cache: !0,
		username: '',
		password: '',
		headers: {},
		xhrFields: {},
		statusCode: {},
		dataType: 'text',
		contentType: 'application/x-www-form-urlencoded',
		timeout: 0,
		global: !0,
	};
	return (
		N(tn, (n, i) => {
			[
				'beforeSend',
				'success',
				'error',
				'complete',
				'statusCode',
			].indexOf(n) < 0 &&
				!z(i) &&
				(e[n] = i);
		}),
		X({}, e, t)
	);
}
function Ca(t) {
	let e = !1;
	const n = {},
		i = ya(t);
	let s = i.url || window.location.toString();
	const o = i.method.toUpperCase();
	let r = i.data;
	const l = i.processData,
		c = i.async,
		f = i.cache,
		h = i.username,
		m = i.password,
		p = i.headers,
		C = i.xhrFields,
		k = i.statusCode,
		O = i.dataType,
		F = i.contentType,
		L = i.timeout,
		q = i.global;
	r &&
		(yn(o) || l) &&
		!wt(r) &&
		!(r instanceof ArrayBuffer) &&
		!(r instanceof Blob) &&
		!(r instanceof Document) &&
		!(r instanceof FormData) &&
		(r = os(r)),
		r && yn(o) && ((s = Qs(s, r)), (r = null));
	function D(H, lt, B, ...x) {
		q && a(document).trigger(H, lt);
		let et, U;
		B &&
			(B in tn && (et = tn[B](...x)),
			i[B] && (U = i[B](...x)),
			B === 'beforeSend' && (et === !1 || U === !1) && (e = !0));
	}
	function V() {
		let H;
		return new Promise((lt, B) => {
			yn(o) && !f && (s = Qs(s, `_=${Date.now()}`));
			const x = new XMLHttpRequest();
			x.open(o, s, c, h, m),
				(F || (r && !yn(o) && F !== !1)) &&
					x.setRequestHeader('Content-Type', F),
				O === 'json' &&
					x.setRequestHeader(
						'Accept',
						'application/json, text/javascript'
					),
				p &&
					N(p, (Q, T) => {
						z(T) || x.setRequestHeader(Q, T + '');
					}),
				(/^([\w-]+:)?\/\/([^/]+)/.test(s) &&
					RegExp.$2 !== window.location.host) ||
					x.setRequestHeader('X-Requested-With', 'XMLHttpRequest'),
				C &&
					N(C, (Q, T) => {
						x[Q] = T;
					}),
				(n.xhr = x),
				(n.options = i);
			let U;
			if (
				((x.onload = function () {
					U && clearTimeout(U);
					const Q =
						(x.status >= 200 && x.status < 300) ||
						x.status === 304 ||
						x.status === 0;
					let T;
					if (Q)
						if (
							(x.status === 204 || o === 'HEAD'
								? (H = 'nocontent')
								: x.status === 304
								? (H = 'notmodified')
								: (H = 'success'),
							O === 'json')
						) {
							try {
								(T =
									o === 'HEAD'
										? void 0
										: JSON.parse(x.responseText)),
									(n.data = T);
							} catch {
								(H = 'parsererror'),
									D(Mt.ajaxError, n, 'error', x, H),
									B(new Error(H));
							}
							H !== 'parsererror' &&
								(D(Mt.ajaxSuccess, n, 'success', T, H, x),
								lt(T));
						} else
							(T =
								o === 'HEAD'
									? void 0
									: x.responseType === 'text' ||
									  x.responseType === ''
									? x.responseText
									: x.response),
								(n.data = T),
								D(Mt.ajaxSuccess, n, 'success', T, H, x),
								lt(T);
					else
						(H = 'error'),
							D(Mt.ajaxError, n, 'error', x, H),
							B(new Error(H));
					N([tn.statusCode, k], (st, ft) => {
						ft &&
							ft[x.status] &&
							(Q ? ft[x.status](T, H, x) : ft[x.status](x, H));
					}),
						D(Mt.ajaxComplete, n, 'complete', x, H);
				}),
				(x.onerror = function () {
					U && clearTimeout(U),
						D(Mt.ajaxError, n, 'error', x, x.statusText),
						D(Mt.ajaxComplete, n, 'complete', x, 'error'),
						B(new Error(x.statusText));
				}),
				(x.onabort = function () {
					let Q = 'abort';
					U && ((Q = 'timeout'), clearTimeout(U)),
						D(Mt.ajaxError, n, 'error', x, Q),
						D(Mt.ajaxComplete, n, 'complete', x, Q),
						B(new Error(Q));
				}),
				D(Mt.ajaxStart, n, 'beforeSend', x),
				e)
			) {
				B(new Error('cancel'));
				return;
			}
			L > 0 &&
				(U = setTimeout(() => {
					x.abort();
				}, L)),
				x.send(r);
		});
	}
	return V();
}
a.ajax = Ca;
function wa(t) {
	return X(tn, t);
}
a.ajaxSetup = wa;
a.contains = Yt;
const Ct = '_mduiElementDataStorage';
function Zs(t, e) {
	t[Ct] || (t[Ct] = {}),
		N(e, (n, i) => {
			t[Ct][hn(n)] = i;
		});
}
function Ce(t, e, n) {
	if (Jt(e)) return Zs(t, e), e;
	if (!z(n)) return Zs(t, { [e]: n }), n;
	if (z(e)) return t[Ct] ? t[Ct] : {};
	if (((e = hn(e)), t[Ct] && e in t[Ct])) return t[Ct][e];
}
a.data = Ce;
a.each = N;
a.extend = function (...t) {
	return t.length === 1
		? (N(t[0], (e, n) => {
				this[e] = n;
		  }),
		  this)
		: X(t.shift(), t.shift(), ...t);
};
function en(t, e) {
	let n;
	const i = [];
	return (
		N(t, (s, o) => {
			(n = e.call(window, o, s)), n != null && i.push(n);
		}),
		[].concat(...i)
	);
}
a.map = en;
a.merge = is;
a.param = os;
function _r(t, e) {
	if (!t[Ct]) return;
	const n = (i) => {
		(i = hn(i)), t[Ct][i] && ((t[Ct][i] = null), delete t[Ct][i]);
	};
	z(e)
		? ((t[Ct] = null), delete t[Ct])
		: wt(e)
		? e
				.split(' ')
				.filter((i) => i)
				.forEach((i) => n(i))
		: N(e, (i, s) => n(s));
}
a.removeData = _r;
function ni(t) {
	const e = [];
	return (
		N(t, (n, i) => {
			e.indexOf(i) === -1 && e.push(i);
		}),
		e
	);
}
a.unique = ni;
a.fn.add = function (t) {
	return new ht(ni(is(this.get(), a(t).get())));
};
N(['add', 'remove', 'toggle'], (t, e) => {
	a.fn[`${e}Class`] = function (n) {
		return e === 'remove' && !arguments.length
			? this.each((i, s) => {
					s.setAttribute('class', '');
			  })
			: this.each((i, s) => {
					if (!qe(s)) return;
					const o = (
						_t(n) ? n.call(s, i, s.getAttribute('class') || '') : n
					)
						.split(' ')
						.filter((r) => r);
					N(o, (r, l) => {
						s.classList[e](l);
					});
			  });
	};
});
N(['insertBefore', 'insertAfter'], (t, e) => {
	a.fn[e] = function (n) {
		const i = t ? a(this.get().reverse()) : this,
			s = a(n),
			o = [];
		return (
			s.each((r, l) => {
				!l.parentNode ||
					i.each((c, f) => {
						const h = r ? f.cloneNode(!0) : f,
							m = t ? l.nextSibling : l;
						o.push(h), l.parentNode.insertBefore(h, m);
					});
			}),
			a(t ? o.reverse() : o)
		);
	};
});
function Ea(t) {
	return wt(t) && (t[0] !== '<' || t[t.length - 1] !== '>');
}
N(['before', 'after'], (t, e) => {
	a.fn[e] = function (...n) {
		return (
			t === 1 && (n = n.reverse()),
			this.each((i, s) => {
				const o = _t(n[0]) ? [n[0].call(s, i, s.innerHTML)] : n;
				N(o, (r, l) => {
					let c;
					Ea(l)
						? (c = a(gr(l, 'div')))
						: i && qe(l)
						? (c = a(l.cloneNode(!0)))
						: (c = a(l)),
						c[t ? 'insertAfter' : 'insertBefore'](s);
				});
			})
		);
	};
});
a.fn.off = function (t, e, n) {
	return Jt(t)
		? (N(t, (i, s) => {
				this.off(i, e, s);
		  }),
		  this)
		: ((e === !1 || _t(e)) && ((n = e), (e = void 0)),
		  n === !1 && (n = br),
		  this.each(function () {
				xa(this, t, n, e);
		  }));
};
a.fn.on = function (t, e, n, i, s) {
	if (Jt(t))
		return (
			wt(e) || ((n = n || e), (e = void 0)),
			N(t, (o, r) => {
				this.on(o, e, n, r, s);
			}),
			this
		);
	if (
		(n == null && i == null
			? ((i = e), (n = e = void 0))
			: i == null &&
			  (wt(e)
					? ((i = n), (n = void 0))
					: ((i = n), (n = e), (e = void 0))),
		i === !1)
	)
		i = br;
	else if (!i) return this;
	if (s) {
		const o = this,
			r = i;
		i = function (l) {
			return o.off(l.type, e, i), r.apply(this, arguments);
		};
	}
	return this.each(function () {
		$a(this, t, i, n, e);
	});
};
N(Mt, (t, e) => {
	a.fn[t] = function (n) {
		return this.on(e, (i, s) => {
			n(i, s.xhr, s.options, s.data);
		});
	};
});
a.fn.map = function (t) {
	return new ht(en(this, (e, n) => t.call(e, n, e)));
};
a.fn.clone = function () {
	return this.map(function () {
		return this.cloneNode(!0);
	});
};
a.fn.is = function (t) {
	let e = !1;
	if (_t(t))
		return (
			this.each((i, s) => {
				t.call(s, i, s) && (e = !0);
			}),
			e
		);
	if (wt(t))
		return (
			this.each((i, s) => {
				if (ns(s) || es(s)) return;
				(s.matches || s.msMatchesSelector).call(s, t) && (e = !0);
			}),
			e
		);
	const n = a(t);
	return (
		this.each((i, s) => {
			n.each((o, r) => {
				s === r && (e = !0);
			});
		}),
		e
	);
};
a.fn.remove = function (t) {
	return this.each((e, n) => {
		n.parentNode && (!t || a(n).is(t)) && n.parentNode.removeChild(n);
	});
};
N(['prepend', 'append'], (t, e) => {
	a.fn[e] = function (...n) {
		return this.each((i, s) => {
			const o = s.childNodes,
				r = o.length,
				l = r ? o[t ? r - 1 : 0] : document.createElement('div');
			r || s.appendChild(l);
			let c = _t(n[0]) ? [n[0].call(s, i, s.innerHTML)] : n;
			i && (c = c.map((f) => (wt(f) ? f : a(f).clone()))),
				a(l)[t ? 'after' : 'before'](...c),
				r || s.removeChild(l);
		});
	};
});
N(['appendTo', 'prependTo'], (t, e) => {
	a.fn[e] = function (n) {
		const i = [],
			s = a(n).map((r, l) => {
				const c = l.childNodes,
					f = c.length;
				if (f) return c[t ? 0 : f - 1];
				const h = document.createElement('div');
				return l.appendChild(h), i.push(h), h;
			}),
			o = this[t ? 'insertBefore' : 'insertAfter'](s);
		return a(i).remove(), o;
	};
});
N(['attr', 'prop', 'css'], (t, e) => {
	function n(s, o, r) {
		if (!z(r))
			switch (t) {
				case 0:
					Ii(r) ? s.removeAttribute(o) : s.setAttribute(o, r);
					break;
				case 1:
					s[o] = r;
					break;
				default:
					(o = hn(o)),
						(s.style[o] = We(r)
							? `${r}${ga.indexOf(o) > -1 ? '' : 'px'}`
							: r);
					break;
			}
	}
	function i(s, o) {
		switch (t) {
			case 0:
				const r = s.getAttribute(o);
				return Ii(r) ? void 0 : r;
			case 1:
				return s[o];
			default:
				return ei(s, o);
		}
	}
	a.fn[e] = function (s, o) {
		if (Jt(s))
			return (
				N(s, (r, l) => {
					this[e](r, l);
				}),
				this
			);
		if (arguments.length === 1) {
			const r = this[0];
			return qe(r) ? i(r, s) : void 0;
		}
		return this.each((r, l) => {
			n(l, s, _t(o) ? o.call(l, r, i(l, s)) : o);
		});
	};
});
a.fn.children = function (t) {
	const e = [];
	return (
		this.each((n, i) => {
			N(i.childNodes, (s, o) => {
				!qe(o) || ((!t || a(o).is(t)) && e.push(o));
			});
		}),
		new ht(ni(e))
	);
};
a.fn.slice = function (...t) {
	return new ht([].slice.apply(this, t));
};
a.fn.eq = function (t) {
	const e = t === -1 ? this.slice(t) : this.slice(t, +t + 1);
	return new ht(e);
};
function rs(t, e, n, i, s) {
	const o = [];
	let r;
	return (
		t.each((l, c) => {
			for (r = c[n]; r && qe(r); ) {
				if (e === 2) {
					if (i && a(r).is(i)) break;
					(!s || a(r).is(s)) && o.push(r);
				} else if (e === 0) {
					(!i || a(r).is(i)) && o.push(r);
					break;
				} else (!i || a(r).is(i)) && o.push(r);
				r = r[n];
			}
		}),
		new ht(ni(o))
	);
}
N(['', 's', 'sUntil'], (t, e) => {
	a.fn[`parent${e}`] = function (n, i) {
		const s = t ? a(this.get().reverse()) : this;
		return rs(s, t, 'parentNode', n, i);
	};
});
a.fn.closest = function (t) {
	if (this.is(t)) return this;
	const e = [];
	return (
		this.parents().each((n, i) => {
			if (a(i).is(t)) return e.push(i), !1;
		}),
		new ht(e)
	);
};
const Ta = /^(?:{[\w\W]*\}|\[[\w\W]*\])$/;
function Oa(t) {
	return t === 'true'
		? !0
		: t === 'false'
		? !1
		: t === 'null'
		? null
		: t === +t + ''
		? +t
		: Ta.test(t)
		? JSON.parse(t)
		: t;
}
function Gs(t, e, n) {
	if (z(n) && t.nodeType === 1) {
		const i = 'data-' + pr(e);
		if (((n = t.getAttribute(i)), wt(n)))
			try {
				n = Oa(n);
			} catch {}
		else n = void 0;
	}
	return n;
}
a.fn.data = function (t, e) {
	if (z(t)) {
		if (!this.length) return;
		const n = this[0],
			i = Ce(n);
		if (n.nodeType !== 1) return i;
		const s = n.attributes;
		let o = s.length;
		for (; o--; )
			if (s[o]) {
				let r = s[o].name;
				r.indexOf('data-') === 0 &&
					((r = hn(r.slice(5))), (i[r] = Gs(n, r, i[r])));
			}
		return i;
	}
	if (Jt(t))
		return this.each(function () {
			Ce(this, t);
		});
	if (arguments.length === 2 && z(e)) return this;
	if (!z(e))
		return this.each(function () {
			Ce(this, t, e);
		});
	if (!!this.length) return Gs(this[0], t, Ce(this[0], t));
};
a.fn.empty = function () {
	return this.each(function () {
		this.innerHTML = '';
	});
};
a.fn.extend = function (t) {
	return (
		N(t, (e, n) => {
			a.fn[e] = n;
		}),
		this
	);
};
a.fn.filter = function (t) {
	if (_t(t)) return this.map((n, i) => (t.call(i, n, i) ? i : void 0));
	if (wt(t)) return this.map((n, i) => (a(i).is(t) ? i : void 0));
	const e = a(t);
	return this.map((n, i) => (e.get().indexOf(i) > -1 ? i : void 0));
};
a.fn.first = function () {
	return this.eq(0);
};
a.fn.has = function (t) {
	const e = wt(t) ? this.find(t) : a(t),
		{ length: n } = e;
	return this.map(function () {
		for (let i = 0; i < n; i += 1) if (Yt(this, e[i])) return this;
	});
};
a.fn.hasClass = function (t) {
	return this[0].classList.contains(t);
};
function $r(t, e, n, i, s, o) {
	const r = (l) => Mi(t, e.toLowerCase(), l) * o;
	return (
		i === 2 && s && (n += r('margin')),
		mr(t)
			? (ma() && o === 1 && ((n += r('border')), (n += r('padding'))),
			  i === 0 && (n -= r('border')),
			  i === 1 && ((n -= r('border')), (n -= r('padding'))))
			: (i === 0 && (n += r('padding')),
			  i === 2 && ((n += r('border')), (n += r('padding')))),
		n
	);
}
function xr(t, e, n, i) {
	const s = `client${e}`,
		o = `scroll${e}`,
		r = `offset${e}`,
		l = `inner${e}`;
	if (es(t)) return n === 2 ? t[l] : Bn(document)[s];
	if (ns(t)) {
		const f = Bn(t);
		return Math.max(t.body[o], f[o], t.body[r], f[r], f[s]);
	}
	const c = parseFloat(ti(t, e.toLowerCase()) || '0');
	return $r(t, e, c, n, i, 1);
}
function Aa(t, e, n, i, s, o) {
	let r = _t(o) ? o.call(t, e, xr(t, n, i, s)) : o;
	if (r == null) return;
	const l = a(t),
		c = n.toLowerCase();
	if (['auto', 'inherit', ''].indexOf(r) > -1) {
		l.css(c, r);
		return;
	}
	const f = r.toString().replace(/\b[0-9.]*/, ''),
		h = parseFloat(r);
	(r = $r(t, n, h, i, s, -1) + (f || 'px')), l.css(c, r);
}
N(['Width', 'Height'], (t, e) => {
	N([`inner${e}`, e.toLowerCase(), `outer${e}`], (n, i) => {
		a.fn[i] = function (s, o) {
			const r = arguments.length && (n < 2 || !ha(s)),
				l = s === !0 || o === !0;
			return r
				? this.each((c, f) => Aa(f, c, e, n, l, s))
				: this.length
				? xr(this[0], e, n, l)
				: void 0;
		};
	});
});
a.fn.hide = function () {
	return this.each(function () {
		this.style.display = 'none';
	});
};
N(['val', 'html', 'text'], (t, e) => {
	const i = { 0: 'value', 1: 'innerHTML', 2: 'textContent' }[t];
	function s(r) {
		if (t === 2) return en(r, (c) => Bn(c)[i]).join('');
		if (!r.length) return;
		const l = r[0];
		return t === 0 && a(l).is('select[multiple]')
			? en(a(l).find('option:checked'), (c) => c.value)
			: l[i];
	}
	function o(r, l) {
		if (z(l)) {
			if (t !== 0) return;
			l = '';
		}
		t === 1 && qe(l) && (l = l.outerHTML), (r[i] = l);
	}
	a.fn[e] = function (r) {
		return arguments.length
			? this.each((l, c) => {
					const f = _t(r) ? r.call(c, l, s(a(c))) : r;
					t === 0 && Array.isArray(f)
						? a(c).is('select[multiple]')
							? en(
									a(c).find('option'),
									(h) =>
										(h.selected = f.indexOf(h.value) > -1)
							  )
							: (c.checked = f.indexOf(c.value) > -1)
						: o(c, f);
			  })
			: s(this);
	};
});
a.fn.index = function (t) {
	return arguments.length
		? wt(t)
			? a(t).get().indexOf(this[0])
			: this.get().indexOf(a(t)[0])
		: this.eq(0).parent().children().get().indexOf(this[0]);
};
a.fn.last = function () {
	return this.eq(-1);
};
N(['', 'All', 'Until'], (t, e) => {
	a.fn[`next${e}`] = function (n, i) {
		return rs(this, t, 'nextElementSibling', n, i);
	};
});
a.fn.not = function (t) {
	const e = this.filter(t);
	return this.map((n, i) => (e.index(i) > -1 ? void 0 : i));
};
a.fn.offsetParent = function () {
	return this.map(function () {
		let t = this.offsetParent;
		for (; t && a(t).css('position') === 'static'; ) t = t.offsetParent;
		return t || document.documentElement;
	});
};
function Cn(t, e) {
	return parseFloat(t.css(e));
}
a.fn.position = function () {
	if (!this.length) return;
	const t = this.eq(0);
	let e,
		n = { left: 0, top: 0 };
	if (t.css('position') === 'fixed') e = t[0].getBoundingClientRect();
	else {
		e = t.offset();
		const i = t.offsetParent();
		(n = i.offset()),
			(n.top += Cn(i, 'border-top-width')),
			(n.left += Cn(i, 'border-left-width'));
	}
	return {
		top: e.top - n.top - Cn(t, 'margin-top'),
		left: e.left - n.left - Cn(t, 'margin-left'),
	};
};
function yr(t) {
	if (!t.getClientRects().length) return { top: 0, left: 0 };
	const e = t.getBoundingClientRect(),
		n = t.ownerDocument.defaultView;
	return { top: e.top + n.pageYOffset, left: e.left + n.pageXOffset };
}
function Sa(t, e, n) {
	const i = a(t),
		s = i.css('position');
	s === 'static' && i.css('position', 'relative');
	const o = yr(t),
		r = i.css('top'),
		l = i.css('left');
	let c, f;
	if ((s === 'absolute' || s === 'fixed') && (r + l).indexOf('auto') > -1) {
		const p = i.position();
		(c = p.top), (f = p.left);
	} else (c = parseFloat(r)), (f = parseFloat(l));
	const m = _t(e) ? e.call(t, n, X({}, o)) : e;
	i.css({
		top: m.top != null ? m.top - o.top + c : void 0,
		left: m.left != null ? m.left - o.left + f : void 0,
	});
}
a.fn.offset = function (t) {
	return arguments.length
		? this.each(function (e) {
				Sa(this, t, e);
		  })
		: this.length
		? yr(this[0])
		: void 0;
};
a.fn.one = function (t, e, n, i) {
	return this.on(t, e, n, i, !0);
};
N(['', 'All', 'Until'], (t, e) => {
	a.fn[`prev${e}`] = function (n, i) {
		const s = t ? a(this.get().reverse()) : this;
		return rs(s, t, 'previousElementSibling', n, i);
	};
});
a.fn.removeAttr = function (t) {
	const e = t.split(' ').filter((n) => n);
	return this.each(function () {
		N(e, (n, i) => {
			this.removeAttribute(i);
		});
	});
};
a.fn.removeData = function (t) {
	return this.each(function () {
		_r(this, t);
	});
};
a.fn.removeProp = function (t) {
	return this.each(function () {
		try {
			delete this[t];
		} catch {}
	});
};
a.fn.replaceWith = function (t) {
	return (
		this.each((e, n) => {
			let i = t;
			_t(i)
				? (i = i.call(n, e, n.innerHTML))
				: e && !wt(i) && (i = a(i).clone()),
				a(n).before(i);
		}),
		this.remove()
	);
};
a.fn.replaceAll = function (t) {
	return a(t).map(
		(e, n) => (a(n).replaceWith(e ? this.clone() : this), this.get())
	);
};
a.fn.serializeArray = function () {
	const t = [];
	return (
		this.each((e, n) => {
			const i = n instanceof HTMLFormElement ? n.elements : [n];
			a(i).each((s, o) => {
				const r = a(o),
					l = o.type,
					c = o.nodeName.toLowerCase();
				if (
					c !== 'fieldset' &&
					o.name &&
					!o.disabled &&
					['input', 'select', 'textarea', 'keygen'].indexOf(c) > -1 &&
					['submit', 'button', 'image', 'reset', 'file'].indexOf(
						l
					) === -1 &&
					(['radio', 'checkbox'].indexOf(l) === -1 || o.checked)
				) {
					const f = r.val();
					(Array.isArray(f) ? f : [f]).forEach((m) => {
						t.push({ name: o.name, value: m });
					});
				}
			});
		}),
		t
	);
};
a.fn.serialize = function () {
	return os(this.serializeArray());
};
const di = {};
function Ia(t) {
	let e, n;
	return (
		di[t] ||
			((e = document.createElement(t)),
			document.body.appendChild(e),
			(n = ei(e, 'display')),
			e.parentNode.removeChild(e),
			n === 'none' && (n = 'block'),
			(di[t] = n)),
		di[t]
	);
}
a.fn.show = function () {
	return this.each(function () {
		this.style.display === 'none' && (this.style.display = ''),
			ei(this, 'display') === 'none' &&
				(this.style.display = Ia(this.nodeName));
	});
};
a.fn.siblings = function (t) {
	return this.prevAll(t).add(this.nextAll(t));
};
a.fn.toggle = function () {
	return this.each(function () {
		ei(this, 'display') === 'none' ? a(this).show() : a(this).hide();
	});
};
a.fn.reflow = function () {
	return this.each(function () {
		return this.clientLeft;
	});
};
a.fn.transition = function (t) {
	return (
		We(t) && (t = `${t}ms`),
		this.each(function () {
			(this.style.webkitTransitionDuration = t),
				(this.style.transitionDuration = t);
		})
	);
};
a.fn.transitionEnd = function (t) {
	const e = this,
		n = ['webkitTransitionEnd', 'transitionend'];
	function i(s) {
		s.target === this &&
			(t.call(this, s),
			N(n, (o, r) => {
				e.off(r, i);
			}));
	}
	return (
		N(n, (s, o) => {
			e.on(o, i);
		}),
		this
	);
};
a.fn.transformOrigin = function (t) {
	return this.each(function () {
		(this.style.webkitTransformOrigin = t),
			(this.style.transformOrigin = t);
	});
};
a.fn.transform = function (t) {
	return this.each(function () {
		(this.style.webkitTransform = t), (this.style.transform = t);
	});
};
const Cr = {};
function ki(t, e, n, i) {
	let s = Ce(i, '_mdui_mutation');
	s || ((s = []), Ce(i, '_mdui_mutation', s)),
		s.indexOf(t) === -1 && (s.push(t), e.call(i, n, i));
}
a.fn.mutation = function () {
	return this.each((t, e) => {
		const n = a(e);
		N(Cr, (i, s) => {
			n.is(i) && ki(i, s, t, e),
				n.find(i).each((o, r) => {
					ki(i, s, o, r);
				});
		});
	});
};
a.showOverlay = function (t) {
	let e = a('.mdui-overlay');
	e.length
		? (e.data('_overlay_is_deleted', !1), z(t) || e.css('z-index', t))
		: (z(t) && (t = 2e3),
		  (e = a('<div class="mdui-overlay">')
				.appendTo(document.body)
				.reflow()
				.css('z-index', t)));
	let n = e.data('_overlay_level') || 0;
	return e.data('_overlay_level', ++n).addClass('mdui-overlay-show');
};
a.hideOverlay = function (t = !1) {
	const e = a('.mdui-overlay');
	if (!e.length) return;
	let n = t ? 1 : e.data('_overlay_level');
	if (n > 1) {
		e.data('_overlay_level', --n);
		return;
	}
	e.data('_overlay_level', 0)
		.removeClass('mdui-overlay-show')
		.data('_overlay_is_deleted', !0)
		.transitionEnd(() => {
			e.data('_overlay_is_deleted') && e.remove();
		});
};
a.lockScreen = function () {
	const t = a('body'),
		e = t.width();
	let n = t.data('_lockscreen_level') || 0;
	t.addClass('mdui-locked')
		.width(e)
		.data('_lockscreen_level', ++n);
};
a.unlockScreen = function (t = !1) {
	const e = a('body');
	let n = t ? 1 : e.data('_lockscreen_level');
	if (n > 1) {
		e.data('_lockscreen_level', --n);
		return;
	}
	e.data('_lockscreen_level', 0).removeClass('mdui-locked').width('');
};
a.throttle = function (t, e = 16) {
	let n = null;
	return function (...i) {
		Ii(n) &&
			(n = setTimeout(() => {
				t.apply(this, i), (n = null);
			}, e));
	};
};
const hi = {};
a.guid = function (t) {
	if (!z(t) && !z(hi[t])) return hi[t];
	function e() {
		return Math.floor((1 + Math.random()) * 65536)
			.toString(16)
			.substring(1);
	}
	const n =
		'_' +
		e() +
		e() +
		'-' +
		e() +
		'-' +
		e() +
		'-' +
		e() +
		'-' +
		e() +
		e() +
		e();
	return z(t) || (hi[t] = n), n;
};
M.mutation = function (t, e) {
	if (z(t) || z(e)) {
		a(document).mutation();
		return;
	}
	(Cr[t] = e), a(t).each((n, i) => ki(t, e, n, i));
};
function Wt(t, e, n, i, s) {
	s || (s = {}), (s.inst = i);
	const o = `${t}.mdui.${e}`;
	typeof jQuery < 'u' && jQuery(n).trigger(o, s);
	const r = a(n);
	r.trigger(o, s);
	const l = { bubbles: !0, cancelable: !0, detail: s },
		c = new CustomEvent(o, l);
	(c._detail = s), r[0].dispatchEvent(c);
}
const at = a(document),
	ut = a(window);
a('body');
const Ma = {
	tolerance: 5,
	offset: 0,
	initialClass: 'mdui-headroom',
	pinnedClass: 'mdui-headroom-pinned-top',
	unpinnedClass: 'mdui-headroom-unpinned-top',
};
class ka {
	constructor(e, n = {}) {
		(this.options = X({}, Ma)),
			(this.state = 'pinned'),
			(this.isEnable = !1),
			(this.lastScrollY = 0),
			(this.rafId = 0),
			(this.$element = a(e).first()),
			X(this.options, n);
		const i = this.options.tolerance;
		We(i) && (this.options.tolerance = { down: i, up: i }), this.enable();
	}
	onScroll() {
		this.rafId = window.requestAnimationFrame(() => {
			const e = window.pageYOffset,
				n = e > this.lastScrollY ? 'down' : 'up',
				i = this.options.tolerance[n],
				o = Math.abs(e - this.lastScrollY) >= i;
			e > this.lastScrollY && e >= this.options.offset && o
				? this.unpin()
				: ((e < this.lastScrollY && o) || e <= this.options.offset) &&
				  this.pin(),
				(this.lastScrollY = e);
		});
	}
	triggerEvent(e) {
		Wt(e, 'headroom', this.$element, this);
	}
	transitionEnd() {
		this.state === 'pinning' &&
			((this.state = 'pinned'), this.triggerEvent('pinned')),
			this.state === 'unpinning' &&
				((this.state = 'unpinned'), this.triggerEvent('unpinned'));
	}
	pin() {
		this.state === 'pinning' ||
			this.state === 'pinned' ||
			!this.$element.hasClass(this.options.initialClass) ||
			(this.triggerEvent('pin'),
			(this.state = 'pinning'),
			this.$element
				.removeClass(this.options.unpinnedClass)
				.addClass(this.options.pinnedClass)
				.transitionEnd(() => this.transitionEnd()));
	}
	unpin() {
		this.state === 'unpinning' ||
			this.state === 'unpinned' ||
			!this.$element.hasClass(this.options.initialClass) ||
			(this.triggerEvent('unpin'),
			(this.state = 'unpinning'),
			this.$element
				.removeClass(this.options.pinnedClass)
				.addClass(this.options.unpinnedClass)
				.transitionEnd(() => this.transitionEnd()));
	}
	enable() {
		this.isEnable ||
			((this.isEnable = !0),
			(this.state = 'pinned'),
			this.$element
				.addClass(this.options.initialClass)
				.removeClass(this.options.pinnedClass)
				.removeClass(this.options.unpinnedClass),
			(this.lastScrollY = window.pageYOffset),
			ut.on('scroll', () => this.onScroll()));
	}
	disable() {
		!this.isEnable ||
			((this.isEnable = !1),
			this.$element
				.removeClass(this.options.initialClass)
				.removeClass(this.options.pinnedClass)
				.removeClass(this.options.unpinnedClass),
			ut.off('scroll', () => this.onScroll()),
			window.cancelAnimationFrame(this.rafId));
	}
	getState() {
		return this.state;
	}
}
M.Headroom = ka;
function qt(t, e) {
	const n = a(t).attr(e);
	return n
		? new Function(
				'',
				`var json = ${n}; return JSON.parse(JSON.stringify(json));`
		  )()
		: {};
}
const to = 'mdui-headroom';
a(() => {
	M.mutation(`[${to}]`, function () {
		new M.Headroom(this, qt(this, to));
	});
});
const La = { accordion: !1 };
class wr {
	constructor(e, n = {}) {
		this.options = X({}, La);
		const i = `mdui-${this.getNamespace()}-item`;
		(this.classItem = i),
			(this.classItemOpen = `${i}-open`),
			(this.classHeader = `${i}-header`),
			(this.classBody = `${i}-body`),
			(this.$element = a(e).first()),
			X(this.options, n),
			this.bindEvent();
	}
	bindEvent() {
		const e = this;
		this.$element.on('click', `.${this.classHeader}`, function () {
			const i = a(this).parent();
			e.getItems().each((o, r) => {
				i.is(r) && e.toggle(r);
			});
		}),
			this.$element.on(
				'click',
				`[mdui-${this.getNamespace()}-item-close]`,
				function () {
					const i = a(this).parents(`.${e.classItem}`).first();
					e.close(i);
				}
			);
	}
	isOpen(e) {
		return e.hasClass(this.classItemOpen);
	}
	getItems() {
		return this.$element.children(`.${this.classItem}`);
	}
	getItem(e) {
		return We(e) ? this.getItems().eq(e) : a(e).first();
	}
	triggerEvent(e, n) {
		Wt(e, this.getNamespace(), n, this);
	}
	transitionEnd(e, n) {
		this.isOpen(n)
			? (e.transition(0).height('auto').reflow().transition(''),
			  this.triggerEvent('opened', n))
			: (e.height(''), this.triggerEvent('closed', n));
	}
	open(e) {
		const n = this.getItem(e);
		if (this.isOpen(n)) return;
		this.options.accordion &&
			this.$element.children(`.${this.classItemOpen}`).each((s, o) => {
				const r = a(o);
				r.is(n) || this.close(r);
			});
		const i = n.children(`.${this.classBody}`);
		i
			.height(i[0].scrollHeight)
			.transitionEnd(() => this.transitionEnd(i, n)),
			this.triggerEvent('open', n),
			n.addClass(this.classItemOpen);
	}
	close(e) {
		const n = this.getItem(e);
		if (!this.isOpen(n)) return;
		const i = n.children(`.${this.classBody}`);
		this.triggerEvent('close', n),
			n.removeClass(this.classItemOpen),
			i
				.transition(0)
				.height(i[0].scrollHeight)
				.reflow()
				.transition('')
				.height('')
				.transitionEnd(() => this.transitionEnd(i, n));
	}
	toggle(e) {
		const n = this.getItem(e);
		this.isOpen(n) ? this.close(n) : this.open(n);
	}
	openAll() {
		this.getItems().each((e, n) => this.open(n));
	}
	closeAll() {
		this.getItems().each((e, n) => this.close(n));
	}
}
class Pa extends wr {
	getNamespace() {
		return 'collapse';
	}
}
M.Collapse = Pa;
const eo = 'mdui-collapse';
a(() => {
	M.mutation(`[${eo}]`, function () {
		new M.Collapse(this, qt(this, eo));
	});
});
class Da extends wr {
	getNamespace() {
		return 'panel';
	}
}
M.Panel = Da;
const no = 'mdui-panel';
a(() => {
	M.mutation(`[${no}]`, function () {
		new M.Panel(this, qt(this, no));
	});
});
class Er {
	constructor(e) {
		(this.$thRow = a()),
			(this.$tdRows = a()),
			(this.$thCheckbox = a()),
			(this.$tdCheckboxs = a()),
			(this.selectable = !1),
			(this.selectedRow = 0),
			(this.$element = a(e).first()),
			this.init();
	}
	init() {
		(this.$thRow = this.$element.find('thead tr')),
			(this.$tdRows = this.$element.find('tbody tr')),
			(this.selectable = this.$element.hasClass('mdui-table-selectable')),
			this.updateThCheckbox(),
			this.updateTdCheckbox(),
			this.updateNumericCol();
	}
	createCheckboxHTML(e) {
		return `<${e} class="mdui-table-cell-checkbox"><label class="mdui-checkbox"><input type="checkbox"/><i class="mdui-checkbox-icon"></i></label></${e}>`;
	}
	updateThCheckboxStatus() {
		const e = this.$thCheckbox[0],
			n = this.selectedRow,
			i = this.$tdRows.length;
		(e.checked = n === i), (e.indeterminate = !!n && n !== i);
	}
	updateTdCheckbox() {
		const e = 'mdui-table-row-selected';
		this.$tdRows.each((n, i) => {
			const s = a(i);
			if (
				(s.find('.mdui-table-cell-checkbox').remove(), !this.selectable)
			)
				return;
			const o = a(this.createCheckboxHTML('td'))
				.prependTo(s)
				.find('input[type="checkbox"]');
			s.hasClass(e) && ((o[0].checked = !0), this.selectedRow++),
				this.updateThCheckboxStatus(),
				o.on('change', () => {
					o[0].checked
						? (s.addClass(e), this.selectedRow++)
						: (s.removeClass(e), this.selectedRow--),
						this.updateThCheckboxStatus();
				}),
				(this.$tdCheckboxs = this.$tdCheckboxs.add(o));
		});
	}
	updateThCheckbox() {
		this.$thRow.find('.mdui-table-cell-checkbox').remove(),
			this.selectable &&
				(this.$thCheckbox = a(this.createCheckboxHTML('th'))
					.prependTo(this.$thRow)
					.find('input[type="checkbox"]')
					.on('change', () => {
						const e = this.$thCheckbox[0].checked;
						(this.selectedRow = e ? this.$tdRows.length : 0),
							this.$tdCheckboxs.each((n, i) => {
								i.checked = e;
							}),
							this.$tdRows.each((n, i) => {
								e
									? a(i).addClass('mdui-table-row-selected')
									: a(i).removeClass(
											'mdui-table-row-selected'
									  );
							});
					}));
	}
	updateNumericCol() {
		const e = 'mdui-table-col-numeric';
		this.$thRow.find('th').each((n, i) => {
			const s = a(i).hasClass(e);
			this.$tdRows.each((o, r) => {
				const l = a(r).find('td').eq(n);
				s ? l.addClass(e) : l.removeClass(e);
			});
		});
	}
}
const Un = '_mdui_table';
a(() => {
	M.mutation('.mdui-table', function () {
		const t = a(this);
		t.data(Un) || t.data(Un, new Er(t));
	});
});
M.updateTables = function (t) {
	(z(t) ? a('.mdui-table') : a(t)).each((n, i) => {
		const s = a(i),
			o = s.data(Un);
		o ? o.init() : s.data(Un, new Er(s));
	});
};
const Ne = 'touchstart mousedown',
	Tr = 'touchmove mousemove',
	ls = 'touchend mouseup',
	Or = 'touchcancel mouseleave',
	cs = 'touchend touchmove touchcancel';
let Mn = 0;
function fn(t) {
	return !(
		Mn &&
		[
			'mousedown',
			'mouseup',
			'mousemove',
			'click',
			'mouseover',
			'mouseout',
			'mouseenter',
			'mouseleave',
		].indexOf(t.type) > -1
	);
}
function Re(t) {
	t.type === 'touchstart'
		? (Mn += 1)
		: ['touchmove', 'touchend', 'touchcancel'].indexOf(t.type) > -1 &&
		  setTimeout(function () {
				Mn && (Mn -= 1);
		  }, 500);
}
function pi(t, e) {
	if (t instanceof MouseEvent && t.button === 2) return;
	const n =
			typeof TouchEvent < 'u' &&
			t instanceof TouchEvent &&
			t.touches.length
				? t.touches[0]
				: t,
		i = n.pageX,
		s = n.pageY,
		o = e.offset(),
		r = e.innerHeight(),
		l = e.innerWidth(),
		c = { x: i - o.left, y: s - o.top },
		f = Math.max(Math.pow(Math.pow(r, 2) + Math.pow(l, 2), 0.5), 48),
		h = `translate3d(${-c.x + l / 2}px,${-c.y + r / 2}px, 0) scale(1)`;
	a(
		`<div class="mdui-ripple-wave" style="width:${f}px;height:${f}px;margin-top:-${
			f / 2
		}px;margin-left:-${f / 2}px;left:${c.x}px;top:${c.y}px;"></div>`
	)
		.data('_ripple_wave_translate', h)
		.prependTo(e)
		.reflow()
		.transform(h);
}
function Fa(t) {
	if (!t.length || t.data('_ripple_wave_removed')) return;
	t.data('_ripple_wave_removed', !0);
	let e = setTimeout(() => t.remove(), 400);
	const n = t.data('_ripple_wave_translate');
	t.addClass('mdui-ripple-wave-fill')
		.transform(n.replace('scale(1)', 'scale(1.01)'))
		.transitionEnd(() => {
			clearTimeout(e),
				t
					.addClass('mdui-ripple-wave-out')
					.transform(n.replace('scale(1)', 'scale(1.01)')),
				(e = setTimeout(() => t.remove(), 700)),
				setTimeout(() => {
					t.transitionEnd(() => {
						clearTimeout(e), t.remove();
					});
				}, 0);
		});
}
function Li() {
	const t = a(this);
	t.children('.mdui-ripple-wave').each((e, n) => {
		Fa(a(n));
	}),
		t.off(`${Tr} ${ls} ${Or}`, Li);
}
function Ha(t) {
	if (!fn(t) || (Re(t), t.target === document)) return;
	const e = a(t.target),
		n = e.hasClass('mdui-ripple') ? e : e.parents('.mdui-ripple').first();
	if (!!n.length && !(n.prop('disabled') || !z(n.attr('disabled'))))
		if (t.type === 'touchstart') {
			let i = !1,
				s = setTimeout(() => {
					(s = 0), pi(t, n);
				}, 200);
			const o = () => {
					s && (clearTimeout(s), (s = 0), pi(t, n)),
						i || ((i = !0), Li.call(n));
				},
				r = () => {
					s && (clearTimeout(s), (s = 0)), o();
				};
			n.on('touchmove', r).on('touchend touchcancel', o);
		} else pi(t, n), n.on(`${Tr} ${ls} ${Or}`, Li);
}
a(() => {
	at.on(Ne, Ha).on(cs, Re);
});
const Na = { reInit: !1, domLoadedEvent: !1 };
function Ra(t, e = {}) {
	e = X({}, Na, e);
	const n = t.target,
		i = a(n),
		s = t.type,
		o = i.val(),
		r = i.attr('type') || '';
	if (
		['checkbox', 'button', 'submit', 'range', 'radio', 'image'].indexOf(r) >
		-1
	)
		return;
	const l = i.parent('.mdui-textfield');
	if (
		(s === 'focus' && l.addClass('mdui-textfield-focus'),
		s === 'blur' && l.removeClass('mdui-textfield-focus'),
		(s === 'blur' || s === 'input') &&
			(o
				? l.addClass('mdui-textfield-not-empty')
				: l.removeClass('mdui-textfield-not-empty')),
		n.disabled
			? l.addClass('mdui-textfield-disabled')
			: l.removeClass('mdui-textfield-disabled'),
		(s === 'input' || s === 'blur') &&
			!e.domLoadedEvent &&
			n.validity &&
			(n.validity.valid
				? l.removeClass('mdui-textfield-invalid-html5')
				: l.addClass('mdui-textfield-invalid-html5')),
		i.is('textarea'))
	) {
		const f = o;
		let h = !1;
		f.replace(/[\r\n]/g, '') === '' && (i.val(' ' + f), (h = !0)),
			i.outerHeight('');
		const m = i.outerHeight(),
			p = n.scrollHeight;
		p > m && i.outerHeight(p), h && i.val(f);
	}
	e.reInit && l.find('.mdui-textfield-counter').remove();
	const c = i.attr('maxlength');
	c &&
		((e.reInit || e.domLoadedEvent) &&
			a(
				`<div class="mdui-textfield-counter"><span class="mdui-textfield-counter-inputed"></span> / ${c}</div>`
			).appendTo(l),
		l.find('.mdui-textfield-counter-inputed').text(o.length.toString())),
		(l.find('.mdui-textfield-helper').length ||
			l.find('.mdui-textfield-error').length ||
			c) &&
			l.addClass('mdui-textfield-has-bottom');
}
a(() => {
	at.on('input focus blur', '.mdui-textfield-input', { useCapture: !0 }, Ra),
		at.on(
			'click',
			'.mdui-textfield-expandable .mdui-textfield-icon',
			function () {
				a(this)
					.parents('.mdui-textfield')
					.addClass('mdui-textfield-expanded')
					.find('.mdui-textfield-input')[0]
					.focus();
			}
		),
		at.on(
			'click',
			'.mdui-textfield-expanded .mdui-textfield-close',
			function () {
				a(this)
					.parents('.mdui-textfield')
					.removeClass('mdui-textfield-expanded')
					.find('.mdui-textfield-input')
					.val('');
			}
		),
		M.mutation('.mdui-textfield', function () {
			a(this)
				.find('.mdui-textfield-input')
				.trigger('input', { domLoadedEvent: !0 });
		});
});
M.updateTextFields = function (t) {
	(z(t) ? a('.mdui-textfield') : a(t)).each((n, i) => {
		a(i).find('.mdui-textfield-input').trigger('input', { reInit: !0 });
	});
};
function Ar(t) {
	const e = t.data(),
		n = e._slider_$track,
		i = e._slider_$fill,
		s = e._slider_$thumb,
		o = e._slider_$input,
		r = e._slider_min,
		l = e._slider_max,
		c = e._slider_disabled,
		f = e._slider_discrete,
		h = e._slider_$thumbText,
		m = o.val(),
		p = ((m - r) / (l - r)) * 100;
	i.width(`${p}%`),
		n.width(`${100 - p}%`),
		c && (i.css('padding-right', '6px'), n.css('padding-left', '6px')),
		s.css('left', `${p}%`),
		f && h.text(m),
		p === 0
			? t.addClass('mdui-slider-zero')
			: t.removeClass('mdui-slider-zero');
}
function Sr(t) {
	const e = a('<div class="mdui-slider-track"></div>'),
		n = a('<div class="mdui-slider-fill"></div>'),
		i = a('<div class="mdui-slider-thumb"></div>'),
		s = t.find('input[type="range"]'),
		o = s[0].disabled,
		r = t.hasClass('mdui-slider-discrete');
	o
		? t.addClass('mdui-slider-disabled')
		: t.removeClass('mdui-slider-disabled'),
		t.find('.mdui-slider-track').remove(),
		t.find('.mdui-slider-fill').remove(),
		t.find('.mdui-slider-thumb').remove(),
		t.append(e).append(n).append(i);
	let l = a();
	r && ((l = a('<span></span>')), i.empty().append(l)),
		t.data('_slider_$track', e),
		t.data('_slider_$fill', n),
		t.data('_slider_$thumb', i),
		t.data('_slider_$input', s),
		t.data('_slider_min', s.attr('min')),
		t.data('_slider_max', s.attr('max')),
		t.data('_slider_disabled', o),
		t.data('_slider_discrete', r),
		t.data('_slider_$thumbText', l),
		Ar(t);
}
const wn = '.mdui-slider input[type="range"]';
a(() => {
	at.on('input change', wn, function () {
		const t = a(this).parent();
		Ar(t);
	}),
		at.on(Ne, wn, function (t) {
			if (!fn(t) || (Re(t), this.disabled)) return;
			a(this).parent().addClass('mdui-slider-focus');
		}),
		at.on(ls, wn, function (t) {
			if (!fn(t) || this.disabled) return;
			a(this).parent().removeClass('mdui-slider-focus');
		}),
		at.on(cs, wn, Re),
		M.mutation('.mdui-slider', function () {
			Sr(a(this));
		});
});
M.updateSliders = function (t) {
	(z(t) ? a('.mdui-slider') : a(t)).each((n, i) => {
		Sr(a(i));
	});
};
const ja = { trigger: 'hover' };
class Ba {
	constructor(e, n = {}) {
		(this.options = X({}, ja)),
			(this.state = 'closed'),
			(this.$element = a(e).first()),
			X(this.options, n),
			(this.$btn = this.$element.find('.mdui-fab')),
			(this.$dial = this.$element.find('.mdui-fab-dial')),
			(this.$dialBtns = this.$dial.find('.mdui-fab')),
			this.options.trigger === 'hover' &&
				(this.$btn.on('touchstart mouseenter', () => this.open()),
				this.$element.on('mouseleave', () => this.close())),
			this.options.trigger === 'click' &&
				this.$btn.on(Ne, () => this.open()),
			at.on(Ne, (i) => {
				a(i.target).parents('.mdui-fab-wrapper').length || this.close();
			});
	}
	triggerEvent(e) {
		Wt(e, 'fab', this.$element, this);
	}
	isOpen() {
		return this.state === 'opening' || this.state === 'opened';
	}
	open() {
		this.isOpen() ||
			(this.$dialBtns.each((e, n) => {
				const i = `${15 * (this.$dialBtns.length - e)}ms`;
				(n.style.transitionDelay = i),
					(n.style.webkitTransitionDelay = i);
			}),
			this.$dial.css('height', 'auto').addClass('mdui-fab-dial-show'),
			this.$btn.find('.mdui-fab-opened').length &&
				this.$btn.addClass('mdui-fab-opened'),
			(this.state = 'opening'),
			this.triggerEvent('open'),
			this.$dialBtns.first().transitionEnd(() => {
				this.$btn.hasClass('mdui-fab-opened') &&
					((this.state = 'opened'), this.triggerEvent('opened'));
			}));
	}
	close() {
		!this.isOpen() ||
			(this.$dialBtns.each((e, n) => {
				const i = `${15 * e}ms`;
				(n.style.transitionDelay = i),
					(n.style.webkitTransitionDelay = i);
			}),
			this.$dial.removeClass('mdui-fab-dial-show'),
			this.$btn.removeClass('mdui-fab-opened'),
			(this.state = 'closing'),
			this.triggerEvent('close'),
			this.$dialBtns.last().transitionEnd(() => {
				this.$btn.hasClass('mdui-fab-opened') ||
					((this.state = 'closed'),
					this.triggerEvent('closed'),
					this.$dial.css('height', 0));
			}));
	}
	toggle() {
		this.isOpen() ? this.close() : this.open();
	}
	show() {
		this.$element.removeClass('mdui-fab-hide');
	}
	hide() {
		this.$element.addClass('mdui-fab-hide');
	}
	getState() {
		return this.state;
	}
}
M.Fab = Ba;
const io = 'mdui-fab';
a(() => {
	at.on('touchstart mousedown mouseover', `[${io}]`, function () {
		new M.Fab(this, qt(this, io));
	});
});
const Ua = { position: 'auto', gutter: 16 };
class Wa {
	constructor(e, n = {}) {
		(this.$element = a()),
			(this.options = X({}, Ua)),
			(this.size = 0),
			(this.$selected = a()),
			(this.$menu = a()),
			(this.$items = a()),
			(this.selectedIndex = 0),
			(this.selectedText = ''),
			(this.selectedValue = ''),
			(this.state = 'closed'),
			(this.$native = a(e).first()),
			this.$native.hide(),
			X(this.options, n),
			(this.uniqueID = a.guid()),
			this.handleUpdate(),
			at.on('click touchstart', (i) => {
				const s = a(i.target);
				this.isOpen() &&
					!s.is(this.$element) &&
					!Yt(this.$element[0], s[0]) &&
					this.close();
			});
	}
	readjustMenu() {
		const e = ut.height(),
			n = this.$element.height(),
			i = this.$items.first(),
			s = i.height(),
			o = parseInt(i.css('margin-top')),
			r = this.$element.innerWidth() + 0.01;
		let l = s * this.size + o * 2;
		const c = this.$element[0].getBoundingClientRect().top;
		let f, h;
		if (this.options.position === 'bottom') (h = n), (f = '0px');
		else if (this.options.position === 'top') (h = -l - 1), (f = '100%');
		else {
			const m = e - this.options.gutter * 2;
			l > m && (l = m), (h = -(o + this.selectedIndex * s + (s - n) / 2));
			const p = -(o + (this.size - 1) * s + (s - n) / 2);
			h < p && (h = p);
			const C = c + h;
			C < this.options.gutter
				? (h = -(c - this.options.gutter))
				: C + l + this.options.gutter > e &&
				  (h = -(c + l + this.options.gutter - e)),
				(f = `${this.selectedIndex * s + s / 2 + o}px`);
		}
		this.$element.innerWidth(r),
			this.$menu
				.innerWidth(r)
				.height(l)
				.css({
					'margin-top': h + 'px',
					'transform-origin': 'center ' + f + ' 0',
				});
	}
	isOpen() {
		return this.state === 'opening' || this.state === 'opened';
	}
	handleUpdate() {
		this.isOpen() && this.close(),
			(this.selectedValue = this.$native.val());
		const e = [];
		(this.$items = a()),
			this.$native.find('option').each((i, s) => {
				const o = s.textContent || '',
					r = s.value,
					l = s.disabled,
					c = this.selectedValue === r;
				e.push({
					value: r,
					text: o,
					disabled: l,
					selected: c,
					index: i,
				}),
					c && ((this.selectedText = o), (this.selectedIndex = i)),
					(this.$items = this.$items.add(
						'<div class="mdui-select-menu-item mdui-ripple"' +
							(l ? ' disabled' : '') +
							(c ? ' selected' : '') +
							`>${o}</div>`
					));
			}),
			(this.$selected = a(
				`<span class="mdui-select-selected">${this.selectedText}</span>`
			)),
			(this.$element = a(
				`<div class="mdui-select mdui-select-position-${
					this.options.position
				}" style="${this.$native.attr('style')}" id="${
					this.uniqueID
				}"></div>`
			)
				.show()
				.append(this.$selected)),
			(this.$menu = a('<div class="mdui-select-menu"></div>')
				.appendTo(this.$element)
				.append(this.$items)),
			a(`#${this.uniqueID}`).remove(),
			this.$native.after(this.$element),
			(this.size = parseInt(this.$native.attr('size') || '0')),
			this.size <= 0 &&
				((this.size = this.$items.length),
				this.size > 8 && (this.size = 8));
		const n = this;
		this.$items.on('click', function () {
			if (n.state === 'closing') return;
			const i = a(this),
				s = i.index(),
				o = e[s];
			o.disabled ||
				(n.$selected.text(o.text),
				n.$native.val(o.value),
				n.$items.removeAttr('selected'),
				i.attr('selected', ''),
				(n.selectedIndex = o.index),
				(n.selectedValue = o.value),
				(n.selectedText = o.text),
				n.$native.trigger('change'),
				n.close());
		}),
			this.$element.on('click', (i) => {
				const s = a(i.target);
				s.is('.mdui-select-menu') ||
					s.is('.mdui-select-menu-item') ||
					this.toggle();
			});
	}
	transitionEnd() {
		this.$element.removeClass('mdui-select-closing'),
			this.state === 'opening' &&
				((this.state = 'opened'),
				this.triggerEvent('opened'),
				this.$menu.css('overflow-y', 'auto')),
			this.state === 'closing' &&
				((this.state = 'closed'),
				this.triggerEvent('closed'),
				this.$element.innerWidth(''),
				this.$menu.css({ 'margin-top': '', height: '', width: '' }));
	}
	triggerEvent(e) {
		Wt(e, 'select', this.$native, this);
	}
	toggle() {
		this.isOpen() ? this.close() : this.open();
	}
	open() {
		this.isOpen() ||
			((this.state = 'opening'),
			this.triggerEvent('open'),
			this.readjustMenu(),
			this.$element.addClass('mdui-select-open'),
			this.$menu.transitionEnd(() => this.transitionEnd()));
	}
	close() {
		!this.isOpen() ||
			((this.state = 'closing'),
			this.triggerEvent('close'),
			this.$menu.css('overflow-y', ''),
			this.$element
				.removeClass('mdui-select-open')
				.addClass('mdui-select-closing'),
			this.$menu.transitionEnd(() => this.transitionEnd()));
	}
	getState() {
		return this.state;
	}
}
M.Select = Wa;
const so = 'mdui-select';
a(() => {
	M.mutation(`[${so}]`, function () {
		new M.Select(this, qt(this, so));
	});
});
a(() => {
	M.mutation('.mdui-appbar-scroll-hide', function () {
		new M.Headroom(this);
	}),
		M.mutation('.mdui-appbar-scroll-toolbar-hide', function () {
			new M.Headroom(this, {
				pinnedClass: 'mdui-headroom-pinned-toolbar',
				unpinnedClass: 'mdui-headroom-unpinned-toolbar',
			});
		});
});
const qa = { trigger: 'click', loop: !1 };
class Ka {
	constructor(e, n = {}) {
		(this.options = X({}, qa)),
			(this.activeIndex = -1),
			(this.$element = a(e).first()),
			X(this.options, n),
			(this.$tabs = this.$element.children('a')),
			(this.$indicator = a(
				'<div class="mdui-tab-indicator"></div>'
			).appendTo(this.$element));
		const i = window.location.hash;
		i &&
			this.$tabs.each((s, o) =>
				a(o).attr('href') === i ? ((this.activeIndex = s), !1) : !0
			),
			this.activeIndex === -1 &&
				this.$tabs.each((s, o) =>
					a(o).hasClass('mdui-tab-active')
						? ((this.activeIndex = s), !1)
						: !0
				),
			this.$tabs.length &&
				this.activeIndex === -1 &&
				(this.activeIndex = 0),
			this.setActive(),
			ut.on(
				'resize',
				a.throttle(() => this.setIndicatorPosition(), 100)
			),
			this.$tabs.each((s, o) => {
				this.bindTabEvent(o);
			});
	}
	isDisabled(e) {
		return e.attr('disabled') !== void 0;
	}
	bindTabEvent(e) {
		const n = a(e),
			i = () => {
				if (this.isDisabled(n)) return !1;
				(this.activeIndex = this.$tabs.index(e)), this.setActive();
			};
		n.on('click', i),
			this.options.trigger === 'hover' && n.on('mouseenter', i),
			n.on('click', () => {
				if ((n.attr('href') || '').indexOf('#') === 0) return !1;
			});
	}
	triggerEvent(e, n, i = {}) {
		Wt(e, 'tab', n, this, i);
	}
	setActive() {
		this.$tabs.each((e, n) => {
			const i = a(n),
				s = i.attr('href') || '';
			e === this.activeIndex && !this.isDisabled(i)
				? (i.hasClass('mdui-tab-active') ||
						(this.triggerEvent('change', this.$element, {
							index: this.activeIndex,
							id: s.substr(1),
						}),
						this.triggerEvent('show', i),
						i.addClass('mdui-tab-active')),
				  a(s).show(),
				  this.setIndicatorPosition())
				: (i.removeClass('mdui-tab-active'), a(s).hide());
		});
	}
	setIndicatorPosition() {
		if (this.activeIndex === -1) {
			this.$indicator.css({ left: 0, width: 0 });
			return;
		}
		const e = this.$tabs.eq(this.activeIndex);
		if (this.isDisabled(e)) return;
		const n = e.offset();
		this.$indicator.css({
			left: `${
				n.left +
				this.$element[0].scrollLeft -
				this.$element[0].getBoundingClientRect().left
			}px`,
			width: `${e.innerWidth()}px`,
		});
	}
	next() {
		this.activeIndex !== -1 &&
			(this.$tabs.length > this.activeIndex + 1
				? this.activeIndex++
				: this.options.loop && (this.activeIndex = 0),
			this.setActive());
	}
	prev() {
		this.activeIndex !== -1 &&
			(this.activeIndex > 0
				? this.activeIndex--
				: this.options.loop &&
				  (this.activeIndex = this.$tabs.length - 1),
			this.setActive());
	}
	show(e) {
		this.activeIndex !== -1 &&
			(We(e)
				? (this.activeIndex = e)
				: this.$tabs.each((n, i) => {
						if (i.id === e) return (this.activeIndex = n), !1;
				  }),
			this.setActive());
	}
	handleUpdate() {
		const e = this.$tabs,
			n = this.$element.children('a'),
			i = e.get(),
			s = n.get();
		if (!n.length) {
			(this.activeIndex = -1),
				(this.$tabs = n),
				this.setIndicatorPosition();
			return;
		}
		n.each((o, r) => {
			i.indexOf(r) < 0 &&
				(this.bindTabEvent(r),
				this.activeIndex === -1
					? (this.activeIndex = 0)
					: o <= this.activeIndex && this.activeIndex++);
		}),
			e.each((o, r) => {
				s.indexOf(r) < 0 &&
					(o < this.activeIndex
						? this.activeIndex--
						: o === this.activeIndex && (this.activeIndex = 0));
			}),
			(this.$tabs = n),
			this.setActive();
	}
}
M.Tab = Ka;
const oo = 'mdui-tab';
a(() => {
	M.mutation(`[${oo}]`, function () {
		new M.Tab(this, qt(this, oo));
	});
});
const za = { overlay: !1, swipe: !1 };
class Va {
	constructor(e, n = {}) {
		(this.options = X({}, za)),
			(this.overlay = !1),
			(this.$element = a(e).first()),
			X(this.options, n),
			(this.position = this.$element.hasClass('mdui-drawer-right')
				? 'right'
				: 'left'),
			this.$element.hasClass('mdui-drawer-close')
				? (this.state = 'closed')
				: this.$element.hasClass('mdui-drawer-open')
				? (this.state = 'opened')
				: this.isDesktop()
				? (this.state = 'opened')
				: (this.state = 'closed'),
			ut.on(
				'resize',
				a.throttle(() => {
					this.isDesktop()
						? (this.overlay &&
								!this.options.overlay &&
								(a.hideOverlay(),
								(this.overlay = !1),
								a.unlockScreen()),
						  this.$element.hasClass('mdui-drawer-close') ||
								(this.state = 'opened'))
						: !this.overlay &&
						  this.state === 'opened' &&
						  (this.$element.hasClass('mdui-drawer-open')
								? (a.showOverlay(),
								  (this.overlay = !0),
								  a.lockScreen(),
								  a('.mdui-overlay').one('click', () =>
										this.close()
								  ))
								: (this.state = 'closed'));
				}, 100)
			),
			this.$element.find('[mdui-drawer-close]').each((i, s) => {
				a(s).on('click', () => this.close());
			}),
			this.swipeSupport();
	}
	isDesktop() {
		return ut.width() >= 1024;
	}
	swipeSupport() {
		const e = this;
		let n,
			i,
			s,
			o,
			r = null,
			l = !1;
		const c = a('body'),
			f = 24;
		function h(q) {
			const D = e.position === 'right' ? -1 : 1,
				V = `translate(${-1 * D * q}px, 0) !important;`,
				H = 'initial !important;';
			e.$element.css('cssText', `transform: ${V}; transition: ${H};`);
		}
		function m() {
			(e.$element[0].style.transform = ''),
				(e.$element[0].style.webkitTransform = ''),
				(e.$element[0].style.transition = ''),
				(e.$element[0].style.webkitTransition = '');
		}
		function p() {
			return e.$element.width() + 10;
		}
		function C(q) {
			return Math.min(
				Math.max(r === 'closing' ? o - q : p() + o - q, 0),
				p()
			);
		}
		function k(q) {
			if (r) {
				let D = q.changedTouches[0].pageX;
				e.position === 'right' && (D = c.width() - D);
				const V = C(D) / p();
				l = !1;
				const H = r;
				(r = null),
					H === 'opening'
						? V < 0.92
							? (m(), e.open())
							: m()
						: V > 0.08
						? (m(), e.close())
						: m(),
					a.unlockScreen();
			} else l = !1;
			c.off({ touchmove: O, touchend: k, touchcancel: O });
		}
		function O(q) {
			let D = q.touches[0].pageX;
			e.position === 'right' && (D = c.width() - D);
			const V = q.touches[0].pageY;
			if (r) h(C(D));
			else if (l) {
				const H = Math.abs(D - i),
					lt = Math.abs(V - s),
					B = 8;
				H > B && lt <= B
					? ((o = D),
					  (r = e.state === 'opened' ? 'closing' : 'opening'),
					  a.lockScreen(),
					  h(C(D)))
					: H <= B && lt > B && k();
			}
		}
		function F(q) {
			(i = q.touches[0].pageX),
				e.position === 'right' && (i = c.width() - i),
				(s = q.touches[0].pageY),
				!(e.state !== 'opened' && (i > f || n !== F)) &&
					((l = !0),
					c.on({ touchmove: O, touchend: k, touchcancel: O }));
		}
		function L() {
			n || (c.on('touchstart', F), (n = F));
		}
		this.options.swipe && L();
	}
	triggerEvent(e) {
		Wt(e, 'drawer', this.$element, this);
	}
	transitionEnd() {
		this.$element.hasClass('mdui-drawer-open')
			? ((this.state = 'opened'), this.triggerEvent('opened'))
			: ((this.state = 'closed'), this.triggerEvent('closed'));
	}
	isOpen() {
		return this.state === 'opening' || this.state === 'opened';
	}
	open() {
		this.isOpen() ||
			((this.state = 'opening'),
			this.triggerEvent('open'),
			this.options.overlay ||
				a('body').addClass(`mdui-drawer-body-${this.position}`),
			this.$element
				.removeClass('mdui-drawer-close')
				.addClass('mdui-drawer-open')
				.transitionEnd(() => this.transitionEnd()),
			(!this.isDesktop() || this.options.overlay) &&
				((this.overlay = !0),
				a.showOverlay().one('click', () => this.close()),
				a.lockScreen()));
	}
	close() {
		!this.isOpen() ||
			((this.state = 'closing'),
			this.triggerEvent('close'),
			this.options.overlay ||
				a('body').removeClass(`mdui-drawer-body-${this.position}`),
			this.$element
				.addClass('mdui-drawer-close')
				.removeClass('mdui-drawer-open')
				.transitionEnd(() => this.transitionEnd()),
			this.overlay &&
				(a.hideOverlay(), (this.overlay = !1), a.unlockScreen()));
	}
	toggle() {
		this.isOpen() ? this.close() : this.open();
	}
	getState() {
		return this.state;
	}
}
M.Drawer = Va;
const ro = 'mdui-drawer';
a(() => {
	M.mutation(`[${ro}]`, function () {
		const t = a(this),
			e = qt(this, ro),
			n = e.target;
		delete e.target;
		const i = a(n).first(),
			s = new M.Drawer(i, e);
		t.on('click', () => s.toggle());
	});
});
const we = {};
function be(t, e) {
	if ((z(we[t]) && (we[t] = []), z(e))) return we[t];
	we[t].push(e);
}
function Ir(t) {
	if (z(we[t]) || !we[t].length) return;
	we[t].shift()();
}
const Ya = {
	history: !0,
	overlay: !0,
	modal: !1,
	closeOnEsc: !0,
	closeOnCancel: !0,
	closeOnConfirm: !0,
	destroyOnClosed: !1,
};
let bt = null;
const me = '_mdui_dialog';
let Se = !1,
	zt;
class Xa {
	constructor(e, n = {}) {
		(this.options = X({}, Ya)),
			(this.state = 'closed'),
			(this.append = !1),
			(this.$element = a(e).first()),
			Yt(document.body, this.$element[0]) ||
				((this.append = !0), a('body').append(this.$element)),
			X(this.options, n),
			this.$element.find('[mdui-dialog-cancel]').each((i, s) => {
				a(s).on('click', () => {
					this.triggerEvent('cancel'),
						this.options.closeOnCancel && this.close();
				});
			}),
			this.$element.find('[mdui-dialog-confirm]').each((i, s) => {
				a(s).on('click', () => {
					this.triggerEvent('confirm'),
						this.options.closeOnConfirm && this.close();
				});
			}),
			this.$element.find('[mdui-dialog-close]').each((i, s) => {
				a(s).on('click', () => this.close());
			});
	}
	triggerEvent(e) {
		Wt(e, 'dialog', this.$element, this);
	}
	readjust() {
		if (!bt) return;
		const e = bt.$element,
			n = e.children('.mdui-dialog-title'),
			i = e.children('.mdui-dialog-content'),
			s = e.children('.mdui-dialog-actions');
		e.height(''), i.height('');
		const o = e.height();
		e.css({ top: `${(ut.height() - o) / 2}px`, height: `${o}px` }),
			i.innerHeight(o - (n.innerHeight() || 0) - (s.innerHeight() || 0));
	}
	hashchangeEvent() {
		window.location.hash.substring(1).indexOf('mdui-dialog') < 0 &&
			bt.close(!0);
	}
	overlayClick(e) {
		a(e.target).hasClass('mdui-overlay') && bt && bt.close();
	}
	transitionEnd() {
		this.$element.hasClass('mdui-dialog-open')
			? ((this.state = 'opened'), this.triggerEvent('opened'))
			: ((this.state = 'closed'),
			  this.triggerEvent('closed'),
			  this.$element.hide(),
			  !be(me).length && !bt && Se && (a.unlockScreen(), (Se = !1)),
			  ut.off('resize', a.throttle(this.readjust, 100)),
			  this.options.destroyOnClosed && this.destroy());
	}
	doOpen() {
		if (
			((bt = this),
			Se || (a.lockScreen(), (Se = !0)),
			this.$element.show(),
			this.readjust(),
			ut.on('resize', a.throttle(this.readjust, 100)),
			(this.state = 'opening'),
			this.triggerEvent('open'),
			this.$element
				.addClass('mdui-dialog-open')
				.transitionEnd(() => this.transitionEnd()),
			zt || (zt = a.showOverlay(5100)),
			this.options.modal
				? zt.off('click', this.overlayClick)
				: zt.on('click', this.overlayClick),
			zt.css('opacity', this.options.overlay ? '' : 0),
			this.options.history)
		) {
			let e = window.location.hash.substring(1);
			e.indexOf('mdui-dialog') > -1 &&
				(e = e.replace(/[&?]?mdui-dialog/g, '')),
				e
					? (window.location.hash = `${e}${
							e.indexOf('?') > -1 ? '&' : '?'
					  }mdui-dialog`)
					: (window.location.hash = 'mdui-dialog'),
				ut.on('hashchange', this.hashchangeEvent);
		}
	}
	isOpen() {
		return this.state === 'opening' || this.state === 'opened';
	}
	open() {
		if (!this.isOpen()) {
			if (
				(bt && (bt.state === 'opening' || bt.state === 'opened')) ||
				be(me).length
			) {
				be(me, () => this.doOpen());
				return;
			}
			this.doOpen();
		}
	}
	close(e = !1) {
		setTimeout(() => {
			!this.isOpen() ||
				((bt = null),
				(this.state = 'closing'),
				this.triggerEvent('close'),
				!be(me).length &&
					zt &&
					(a.hideOverlay(),
					(zt = null),
					a('.mdui-overlay').css('z-index', 2e3)),
				this.$element
					.removeClass('mdui-dialog-open')
					.transitionEnd(() => this.transitionEnd()),
				this.options.history &&
					!be(me).length &&
					(e || window.history.back(),
					ut.off('hashchange', this.hashchangeEvent)),
				setTimeout(() => {
					Ir(me);
				}, 100));
		});
	}
	toggle() {
		this.isOpen() ? this.close() : this.open();
	}
	getState() {
		return this.state;
	}
	destroy() {
		this.append && this.$element.remove(),
			!be(me).length &&
				!bt &&
				(zt && (a.hideOverlay(), (zt = null)),
				Se && (a.unlockScreen(), (Se = !1)));
	}
	handleUpdate() {
		this.readjust();
	}
}
at.on('keydown', (t) => {
	bt &&
		bt.options.closeOnEsc &&
		bt.state === 'opened' &&
		t.keyCode === 27 &&
		bt.close();
});
M.Dialog = Xa;
const lo = 'mdui-dialog',
	co = '_mdui_dialog';
a(() => {
	at.on('click', `[${lo}]`, function () {
		const t = qt(this, lo),
			e = t.target;
		delete t.target;
		const n = a(e).first();
		let i = n.data(co);
		i || ((i = new M.Dialog(n, t)), n.data(co, i)), i.open();
	});
});
const Ja = { text: '', bold: !1, close: !0, onClick: () => {} },
	Qa = {
		title: '',
		content: '',
		buttons: [],
		stackedButtons: !1,
		cssClass: '',
		history: !0,
		overlay: !0,
		modal: !1,
		closeOnEsc: !0,
		destroyOnClosed: !0,
		onOpen: () => {},
		onOpened: () => {},
		onClose: () => {},
		onClosed: () => {},
	};
M.dialog = function (t) {
	var e, n;
	(t = X({}, Qa, t)),
		N(t.buttons, (r, l) => {
			t.buttons[r] = X({}, Ja, l);
		});
	let i = '';
	!((e = t.buttons) === null || e === void 0) &&
		e.length &&
		((i = `<div class="mdui-dialog-actions${
			t.stackedButtons ? ' mdui-dialog-actions-stacked' : ''
		}">`),
		N(t.buttons, (r, l) => {
			i += `<a href="javascript:void(0)" class="mdui-btn mdui-ripple mdui-text-color-primary ${
				l.bold ? 'mdui-btn-bold' : ''
			}">${l.text}</a>`;
		}),
		(i += '</div>'));
	const s =
			`<div class="mdui-dialog ${t.cssClass}">` +
			(t.title ? `<div class="mdui-dialog-title">${t.title}</div>` : '') +
			(t.content
				? `<div class="mdui-dialog-content">${t.content}</div>`
				: '') +
			i +
			'</div>',
		o = new M.Dialog(s, {
			history: t.history,
			overlay: t.overlay,
			modal: t.modal,
			closeOnEsc: t.closeOnEsc,
			destroyOnClosed: t.destroyOnClosed,
		});
	return (
		!((n = t.buttons) === null || n === void 0) &&
			n.length &&
			o.$element.find('.mdui-dialog-actions .mdui-btn').each((r, l) => {
				a(l).on('click', () => {
					t.buttons[r].onClick(o), t.buttons[r].close && o.close();
				});
			}),
		o.$element
			.on('open.mdui.dialog', () => {
				t.onOpen(o);
			})
			.on('opened.mdui.dialog', () => {
				t.onOpened(o);
			})
			.on('close.mdui.dialog', () => {
				t.onClose(o);
			})
			.on('closed.mdui.dialog', () => {
				t.onClosed(o);
			}),
		o.open(),
		o
	);
};
const Za = {
	confirmText: 'ok',
	history: !0,
	modal: !1,
	closeOnEsc: !0,
	closeOnConfirm: !0,
};
M.alert = function (t, e, n, i) {
	return (
		_t(e) && ((i = n), (n = e), (e = '')),
		z(n) && (n = () => {}),
		z(i) && (i = {}),
		(i = X({}, Za, i)),
		M.dialog({
			title: e,
			content: t,
			buttons: [
				{
					text: i.confirmText,
					bold: !1,
					close: i.closeOnConfirm,
					onClick: n,
				},
			],
			cssClass: 'mdui-dialog-alert',
			history: i.history,
			modal: i.modal,
			closeOnEsc: i.closeOnEsc,
		})
	);
};
const Ga = {
	confirmText: 'ok',
	cancelText: 'cancel',
	history: !0,
	modal: !1,
	closeOnEsc: !0,
	closeOnCancel: !0,
	closeOnConfirm: !0,
};
M.confirm = function (t, e, n, i, s) {
	return (
		_t(e) && ((s = i), (i = n), (n = e), (e = '')),
		z(n) && (n = () => {}),
		z(i) && (i = () => {}),
		z(s) && (s = {}),
		(s = X({}, Ga, s)),
		M.dialog({
			title: e,
			content: t,
			buttons: [
				{
					text: s.cancelText,
					bold: !1,
					close: s.closeOnCancel,
					onClick: i,
				},
				{
					text: s.confirmText,
					bold: !1,
					close: s.closeOnConfirm,
					onClick: n,
				},
			],
			cssClass: 'mdui-dialog-confirm',
			history: s.history,
			modal: s.modal,
			closeOnEsc: s.closeOnEsc,
		})
	);
};
const tu = {
	confirmText: 'ok',
	cancelText: 'cancel',
	history: !0,
	modal: !1,
	closeOnEsc: !0,
	closeOnCancel: !0,
	closeOnConfirm: !0,
	type: 'text',
	maxlength: 0,
	defaultValue: '',
	confirmOnEnter: !1,
};
M.prompt = function (t, e, n, i, s) {
	_t(e) && ((s = i), (i = n), (n = e), (e = '')),
		z(n) && (n = () => {}),
		z(i) && (i = () => {}),
		z(s) && (s = {}),
		(s = X({}, tu, s));
	const o =
			'<div class="mdui-textfield">' +
			(t ? `<label class="mdui-textfield-label">${t}</label>` : '') +
			(s.type === 'text'
				? `<input class="mdui-textfield-input" type="text" value="${
						s.defaultValue
				  }" ${s.maxlength ? 'maxlength="' + s.maxlength + '"' : ''}/>`
				: '') +
			(s.type === 'textarea'
				? `<textarea class="mdui-textfield-input" ${
						s.maxlength ? 'maxlength="' + s.maxlength + '"' : ''
				  }>${s.defaultValue}</textarea>`
				: '') +
			'</div>',
		r = (c) => {
			const f = c.$element.find('.mdui-textfield-input').val();
			i(f, c);
		},
		l = (c) => {
			const f = c.$element.find('.mdui-textfield-input').val();
			n(f, c);
		};
	return M.dialog({
		title: e,
		content: o,
		buttons: [
			{
				text: s.cancelText,
				bold: !1,
				close: s.closeOnCancel,
				onClick: r,
			},
			{
				text: s.confirmText,
				bold: !1,
				close: s.closeOnConfirm,
				onClick: l,
			},
		],
		cssClass: 'mdui-dialog-prompt',
		history: s.history,
		modal: s.modal,
		closeOnEsc: s.closeOnEsc,
		onOpen: (c) => {
			const f = c.$element.find('.mdui-textfield-input');
			M.updateTextFields(f),
				f[0].focus(),
				s.type !== 'textarea' &&
					s.confirmOnEnter === !0 &&
					f.on('keydown', (h) => {
						if (h.keyCode === 13) {
							const m = c.$element
								.find('.mdui-textfield-input')
								.val();
							return n(m, c), s.closeOnConfirm && c.close(), !1;
						}
					}),
				s.type === 'textarea' && f.on('input', () => c.handleUpdate()),
				s.maxlength && c.handleUpdate();
		},
	});
};
const eu = { position: 'auto', delay: 0, content: '' };
class nu {
	constructor(e, n = {}) {
		(this.options = X({}, eu)),
			(this.state = 'closed'),
			(this.timeoutId = null),
			(this.$target = a(e).first()),
			X(this.options, n),
			(this.$element = a(
				`<div class="mdui-tooltip" id="${a.guid()}">${
					this.options.content
				}</div>`
			).appendTo(document.body));
		const i = this;
		this.$target
			.on('touchstart mouseenter', function (s) {
				i.isDisabled(this) || !fn(s) || (Re(s), i.open());
			})
			.on('touchend mouseleave', function (s) {
				i.isDisabled(this) || !fn(s) || i.close();
			})
			.on(cs, function (s) {
				i.isDisabled(this) || Re(s);
			});
	}
	isDisabled(e) {
		return e.disabled || a(e).attr('disabled') !== void 0;
	}
	isDesktop() {
		return ut.width() > 1024;
	}
	setPosition() {
		let e, n;
		const i = this.$target[0].getBoundingClientRect(),
			s = this.isDesktop() ? 14 : 24,
			o = this.$element[0].offsetWidth,
			r = this.$element[0].offsetHeight;
		let l = this.options.position;
		switch (
			(l === 'auto' &&
				(i.top + i.height + s + r + 2 < ut.height()
					? (l = 'bottom')
					: s + r + 2 < i.top
					? (l = 'top')
					: s + o + 2 < i.left
					? (l = 'left')
					: i.width + s + o + 2 < ut.width() - i.left
					? (l = 'right')
					: (l = 'bottom')),
			l)
		) {
			case 'bottom':
				(e = -1 * (o / 2)),
					(n = i.height / 2 + s),
					this.$element.transformOrigin('top center');
				break;
			case 'top':
				(e = -1 * (o / 2)),
					(n = -1 * (r + i.height / 2 + s)),
					this.$element.transformOrigin('bottom center');
				break;
			case 'left':
				(e = -1 * (o + i.width / 2 + s)),
					(n = -1 * (r / 2)),
					this.$element.transformOrigin('center right');
				break;
			case 'right':
				(e = i.width / 2 + s),
					(n = -1 * (r / 2)),
					this.$element.transformOrigin('center left');
				break;
		}
		const c = this.$target.offset();
		this.$element.css({
			top: `${c.top + i.height / 2}px`,
			left: `${c.left + i.width / 2}px`,
			'margin-left': `${e}px`,
			'margin-top': `${n}px`,
		});
	}
	triggerEvent(e) {
		Wt(e, 'tooltip', this.$target, this);
	}
	transitionEnd() {
		this.$element.hasClass('mdui-tooltip-open')
			? ((this.state = 'opened'), this.triggerEvent('opened'))
			: ((this.state = 'closed'), this.triggerEvent('closed'));
	}
	isOpen() {
		return this.state === 'opening' || this.state === 'opened';
	}
	doOpen() {
		(this.state = 'opening'),
			this.triggerEvent('open'),
			this.$element
				.addClass('mdui-tooltip-open')
				.transitionEnd(() => this.transitionEnd());
	}
	open(e) {
		if (this.isOpen()) return;
		const n = X({}, this.options);
		e && X(this.options, e),
			n.content !== this.options.content &&
				this.$element.html(this.options.content),
			this.setPosition(),
			this.options.delay
				? (this.timeoutId = setTimeout(
						() => this.doOpen(),
						this.options.delay
				  ))
				: ((this.timeoutId = null), this.doOpen());
	}
	close() {
		this.timeoutId &&
			(clearTimeout(this.timeoutId), (this.timeoutId = null)),
			this.isOpen() &&
				((this.state = 'closing'),
				this.triggerEvent('close'),
				this.$element
					.removeClass('mdui-tooltip-open')
					.transitionEnd(() => this.transitionEnd()));
	}
	toggle() {
		this.isOpen() ? this.close() : this.open();
	}
	getState() {
		return this.state;
	}
}
M.Tooltip = nu;
const ao = 'mdui-tooltip',
	uo = '_mdui_tooltip';
a(() => {
	at.on('touchstart mouseover', `[${ao}]`, function () {
		const t = a(this);
		let e = t.data(uo);
		e || ((e = new M.Tooltip(this, qt(this, ao))), t.data(uo, e));
	});
});
const iu = {
	message: '',
	timeout: 4e3,
	position: 'bottom',
	buttonText: '',
	buttonColor: '',
	closeOnButtonClick: !0,
	closeOnOutsideClick: !0,
	onClick: () => {},
	onButtonClick: () => {},
	onOpen: () => {},
	onOpened: () => {},
	onClose: () => {},
	onClosed: () => {},
};
let En = null;
const fo = '_mdui_snackbar';
class su {
	constructor(e) {
		(this.options = X({}, iu)),
			(this.state = 'closed'),
			(this.timeoutId = null),
			X(this.options, e);
		let n = '',
			i = '';
		this.options.buttonColor.indexOf('#') === 0 ||
		this.options.buttonColor.indexOf('rgb') === 0
			? (n = `style="color:${this.options.buttonColor}"`)
			: this.options.buttonColor !== '' &&
			  (i = `mdui-text-color-${this.options.buttonColor}`),
			(this.$element = a(
				`<div class="mdui-snackbar"><div class="mdui-snackbar-text">${this.options.message}</div>` +
					(this.options.buttonText
						? `<a href="javascript:void(0)" class="mdui-snackbar-action mdui-btn mdui-ripple mdui-ripple-white ${i}" ${n}>${this.options.buttonText}</a>`
						: '') +
					'</div>'
			).appendTo(document.body)),
			this.setPosition('close'),
			this.$element
				.reflow()
				.addClass(`mdui-snackbar-${this.options.position}`);
	}
	closeOnOutsideClick(e) {
		const n = a(e.target);
		!n.hasClass('mdui-snackbar') &&
			!n.parents('.mdui-snackbar').length &&
			En.close();
	}
	setPosition(e) {
		const n = this.$element[0].clientHeight,
			i = this.options.position;
		let s, o;
		i === 'bottom' || i === 'top' ? (s = '-50%') : (s = '0'),
			e === 'open'
				? (o = '0')
				: (i === 'bottom' && (o = n),
				  i === 'top' && (o = -n),
				  (i === 'left-top' || i === 'right-top') && (o = -n - 24),
				  (i === 'left-bottom' || i === 'right-bottom') &&
						(o = n + 24)),
			this.$element.transform(`translate(${s},${o}px`);
	}
	open() {
		if (!(this.state === 'opening' || this.state === 'opened')) {
			if (En) {
				be(fo, () => this.open());
				return;
			}
			(En = this),
				(this.state = 'opening'),
				this.options.onOpen(this),
				this.setPosition('open'),
				this.$element.transitionEnd(() => {
					this.state === 'opening' &&
						((this.state = 'opened'),
						this.options.onOpened(this),
						this.options.buttonText &&
							this.$element
								.find('.mdui-snackbar-action')
								.on('click', () => {
									this.options.onButtonClick(this),
										this.options.closeOnButtonClick &&
											this.close();
								}),
						this.$element.on('click', (e) => {
							a(e.target).hasClass('mdui-snackbar-action') ||
								this.options.onClick(this);
						}),
						this.options.closeOnOutsideClick &&
							at.on(Ne, this.closeOnOutsideClick),
						this.options.timeout &&
							(this.timeoutId = setTimeout(
								() => this.close(),
								this.options.timeout
							)));
				});
		}
	}
	close() {
		this.state === 'closing' ||
			this.state === 'closed' ||
			(this.timeoutId && clearTimeout(this.timeoutId),
			this.options.closeOnOutsideClick &&
				at.off(Ne, this.closeOnOutsideClick),
			(this.state = 'closing'),
			this.options.onClose(this),
			this.setPosition('close'),
			this.$element.transitionEnd(() => {
				this.state === 'closing' &&
					((En = null),
					(this.state = 'closed'),
					this.options.onClosed(this),
					this.$element.remove(),
					Ir(fo));
			}));
	}
}
M.snackbar = function (t, e = {}) {
	wt(t) ? (e.message = t) : (e = t);
	const n = new su(e);
	return n.open(), n;
};
a(() => {
	at.on('click', '.mdui-bottom-nav>a', function () {
		const t = a(this),
			e = t.parent();
		e.children('a').each((n, i) => {
			const s = t.is(i);
			s && Wt('change', 'bottomNav', e[0], void 0, { index: n }),
				s
					? a(i).addClass('mdui-bottom-nav-active')
					: a(i).removeClass('mdui-bottom-nav-active');
		});
	}),
		M.mutation('.mdui-bottom-nav-scroll-hide', function () {
			new M.Headroom(this, {
				pinnedClass: 'mdui-headroom-pinned-down',
				unpinnedClass: 'mdui-headroom-unpinned-down',
			});
		});
});
function Xe(t = !1) {
	return `<div class="mdui-spinner-layer ${
		t ? `mdui-spinner-layer-${t}` : ''
	}"><div class="mdui-spinner-circle-clipper mdui-spinner-left"><div class="mdui-spinner-circle"></div></div><div class="mdui-spinner-gap-patch"><div class="mdui-spinner-circle"></div></div><div class="mdui-spinner-circle-clipper mdui-spinner-right"><div class="mdui-spinner-circle"></div></div></div>`;
}
function Mr(t) {
	const e = a(t),
		n = e.hasClass('mdui-spinner-colorful')
			? Xe(1) + Xe(2) + Xe(3) + Xe(4)
			: Xe();
	e.html(n);
}
a(() => {
	M.mutation('.mdui-spinner', function () {
		Mr(this);
	});
});
M.updateSpinners = function (t) {
	(z(t) ? a('.mdui-spinner') : a(t)).each(function () {
		Mr(this);
	});
};
const ou = {
	position: 'auto',
	align: 'auto',
	gutter: 16,
	fixed: !1,
	covered: 'auto',
	subMenuTrigger: 'hover',
	subMenuDelay: 200,
};
class ru {
	constructor(e, n, i = {}) {
		if (
			((this.options = X({}, ou)),
			(this.state = 'closed'),
			(this.$anchor = a(e).first()),
			(this.$element = a(n).first()),
			!this.$anchor.parent().is(this.$element.parent()))
		)
			throw new Error('anchorSelector and menuSelector must be siblings');
		X(this.options, i),
			(this.isCascade = this.$element.hasClass('mdui-menu-cascade')),
			(this.isCovered =
				this.options.covered === 'auto'
					? !this.isCascade
					: this.options.covered),
			this.$anchor.on('click', () => this.toggle()),
			at.on('click touchstart', (o) => {
				const r = a(o.target);
				this.isOpen() &&
					!r.is(this.$element) &&
					!Yt(this.$element[0], r[0]) &&
					!r.is(this.$anchor) &&
					!Yt(this.$anchor[0], r[0]) &&
					this.close();
			});
		const s = this;
		at.on('click', '.mdui-menu-item', function () {
			const o = a(this);
			!o.find('.mdui-menu').length &&
				o.attr('disabled') === void 0 &&
				s.close();
		}),
			this.bindSubMenuEvent(),
			ut.on(
				'resize',
				a.throttle(() => this.readjust(), 100)
			);
	}
	isOpen() {
		return this.state === 'opening' || this.state === 'opened';
	}
	triggerEvent(e) {
		Wt(e, 'menu', this.$element, this);
	}
	readjust() {
		let e, n, i, s;
		const o = ut.height(),
			r = ut.width(),
			l = this.options.gutter,
			c = this.isCovered,
			f = this.options.fixed;
		let h, m;
		const p = this.$element.width(),
			C = this.$element.height(),
			k = this.$anchor[0].getBoundingClientRect(),
			O = k.top,
			F = k.left,
			L = k.height,
			q = k.width,
			D = o - O - L,
			V = r - F - q,
			H = this.$anchor[0].offsetTop,
			lt = this.$anchor[0].offsetLeft;
		if (
			(this.options.position === 'auto'
				? D + (c ? L : 0) > C + l
					? (i = 'bottom')
					: O + (c ? L : 0) > C + l
					? (i = 'top')
					: (i = 'center')
				: (i = this.options.position),
			this.options.align === 'auto'
				? V + q > p + l
					? (s = 'left')
					: F + q > p + l
					? (s = 'right')
					: (s = 'center')
				: (s = this.options.align),
			i === 'bottom')
		)
			(m = '0'), (n = (c ? 0 : L) + (f ? O : H));
		else if (i === 'top')
			(m = '100%'), (n = (c ? L : 0) + (f ? O - C : H - C));
		else {
			m = '50%';
			let B = C;
			this.isCascade ||
				(C + l * 2 > o && ((B = o - l * 2), this.$element.height(B))),
				(n = (o - B) / 2 + (f ? 0 : H - O));
		}
		if ((this.$element.css('top', `${n}px`), s === 'left'))
			(h = '0'), (e = f ? F : lt);
		else if (s === 'right') (h = '100%'), (e = f ? F + q - p : lt + q - p);
		else {
			h = '50%';
			let B = p;
			p + l * 2 > r && ((B = r - l * 2), this.$element.width(B)),
				(e = (r - B) / 2 + (f ? 0 : lt - F));
		}
		this.$element.css('left', `${e}px`),
			this.$element.transformOrigin(`${h} ${m}`);
	}
	readjustSubmenu(e) {
		const n = e.parent('.mdui-menu-item');
		let i, s, o, r;
		const l = ut.height(),
			c = ut.width();
		let f, h;
		const m = e.width(),
			p = e.height(),
			C = n[0].getBoundingClientRect(),
			k = C.width,
			O = C.height,
			F = C.left,
			L = C.top;
		l - L > p ? (o = 'bottom') : L + O > p ? (o = 'top') : (o = 'bottom'),
			c - F - k > m ? (r = 'left') : F > m ? (r = 'right') : (r = 'left'),
			o === 'bottom'
				? ((h = '0'), (i = '0'))
				: o === 'top' && ((h = '100%'), (i = -p + O)),
			e.css('top', `${i}px`),
			r === 'left'
				? ((f = '0'), (s = k))
				: r === 'right' && ((f = '100%'), (s = -m)),
			e.css('left', `${s}px`),
			e.transformOrigin(`${f} ${h}`);
	}
	openSubMenu(e) {
		this.readjustSubmenu(e),
			e
				.addClass('mdui-menu-open')
				.parent('.mdui-menu-item')
				.addClass('mdui-menu-item-active');
	}
	closeSubMenu(e) {
		e
			.removeClass('mdui-menu-open')
			.addClass('mdui-menu-closing')
			.transitionEnd(() => e.removeClass('mdui-menu-closing'))
			.parent('.mdui-menu-item')
			.removeClass('mdui-menu-item-active'),
			e.find('.mdui-menu').each((n, i) => {
				const s = a(i);
				s.removeClass('mdui-menu-open')
					.addClass('mdui-menu-closing')
					.transitionEnd(() => s.removeClass('mdui-menu-closing'))
					.parent('.mdui-menu-item')
					.removeClass('mdui-menu-item-active');
			});
	}
	toggleSubMenu(e) {
		e.hasClass('mdui-menu-open')
			? this.closeSubMenu(e)
			: this.openSubMenu(e);
	}
	bindSubMenuEvent() {
		const e = this;
		if (
			(this.$element.on('click', '.mdui-menu-item', function (n) {
				const i = a(this),
					s = a(n.target);
				if (
					i.attr('disabled') !== void 0 ||
					s.is('.mdui-menu') ||
					s.is('.mdui-divider') ||
					!s.parents('.mdui-menu-item').first().is(i)
				)
					return;
				const o = i.children('.mdui-menu');
				i
					.parent('.mdui-menu')
					.children('.mdui-menu-item')
					.each((r, l) => {
						const c = a(l).children('.mdui-menu');
						c.length &&
							(!o.length || !c.is(o)) &&
							e.closeSubMenu(c);
					}),
					o.length && e.toggleSubMenu(o);
			}),
			this.options.subMenuTrigger === 'hover')
		) {
			let n = null,
				i = null;
			this.$element.on(
				'mouseover mouseout',
				'.mdui-menu-item',
				function (s) {
					const o = a(this),
						r = s.type,
						l = a(s.relatedTarget);
					if (o.attr('disabled') !== void 0) return;
					if (r === 'mouseover') {
						if (!o.is(l) && Yt(o[0], l[0])) return;
					} else if (r === 'mouseout' && (o.is(l) || Yt(o[0], l[0])))
						return;
					const c = o.children('.mdui-menu');
					if (r === 'mouseover') {
						if (c.length) {
							const f = c.data('timeoutClose.mdui.menu');
							if (
								(f && clearTimeout(f),
								c.hasClass('mdui-menu-open'))
							)
								return;
							clearTimeout(i),
								(n = i =
									setTimeout(
										() => e.openSubMenu(c),
										e.options.subMenuDelay
									)),
								c.data('timeoutOpen.mdui.menu', n);
						}
					} else if (r === 'mouseout' && c.length) {
						const f = c.data('timeoutOpen.mdui.menu');
						f && clearTimeout(f),
							(n = setTimeout(
								() => e.closeSubMenu(c),
								e.options.subMenuDelay
							)),
							c.data('timeoutClose.mdui.menu', n);
					}
				}
			);
		}
	}
	transitionEnd() {
		this.$element.removeClass('mdui-menu-closing'),
			this.state === 'opening' &&
				((this.state = 'opened'), this.triggerEvent('opened')),
			this.state === 'closing' &&
				((this.state = 'closed'),
				this.triggerEvent('closed'),
				this.$element.css({
					top: '',
					left: '',
					width: '',
					position: 'fixed',
				}));
	}
	toggle() {
		this.isOpen() ? this.close() : this.open();
	}
	open() {
		this.isOpen() ||
			((this.state = 'opening'),
			this.triggerEvent('open'),
			this.readjust(),
			this.$element
				.css('position', this.options.fixed ? 'fixed' : 'absolute')
				.addClass('mdui-menu-open')
				.transitionEnd(() => this.transitionEnd()));
	}
	close() {
		!this.isOpen() ||
			((this.state = 'closing'),
			this.triggerEvent('close'),
			this.$element.find('.mdui-menu').each((e, n) => {
				this.closeSubMenu(a(n));
			}),
			this.$element
				.removeClass('mdui-menu-open')
				.addClass('mdui-menu-closing')
				.transitionEnd(() => this.transitionEnd()));
	}
}
M.Menu = ru;
const ho = 'mdui-menu',
	po = '_mdui_menu';
a(() => {
	at.on('click', `[${ho}]`, function () {
		const t = a(this);
		let e = t.data(po);
		if (!e) {
			const n = qt(this, ho),
				i = n.target;
			delete n.target,
				(e = new M.Menu(t, i, n)),
				t.data(po, e),
				e.toggle();
		}
	});
});
const lu = (t, e) => {
		const n = t.__vccOpts || t;
		for (const [i, s] of e) n[i] = s;
		return n;
	},
	cu = { class: 'mock' },
	au = { class: 'main' },
	uu = {
		__name: 'loading',
		props: { loading: Boolean },
		setup(t) {
			return (e, n) => (
				Hn(),
				Nn('section', null, [
					K('div', cu, [
						K(
							'div',
							au,
							Je(
								t.loading
									? '\u5C11\u5973\u7948\u7977\u4E2D...'
									: '\u52A0\u8F7D\u5B8C\u6BD5'
							),
							1
						),
					]),
				])
			);
		},
	},
	fu = lu(uu, [['__scopeId', 'data-v-a5173233']]);
const du = { class: 'mdui-theme-primary-indigo mdui-theme-accent-pink' },
	hu = wc(
		'<link rel="stylesheet" href="https://unpkg.com/mdui@1.0.2/dist/css/mdui.min.css"><div class="mdui-toolbar mdui-color-theme"><a href="javascript:;" class="drag mdui-btn mdui-btn-icon"><i class="mdui-icon material-icons">menu</i></a><span class="mdui-typo-title">\u539F\u795E\u8BED\u97F3\u751F\u6210\u5668</span><div class="mdui-toolbar-spacer"></div><a href="javascript:location.reload();;" class="mdui-btn mdui-btn-icon"><i class="mdui-icon material-icons">refresh</i></a></div>',
		2
	),
	pu = { class: 'mdui-container' },
	mu = { class: 'mdui-textfield mdui-textfield-floating-label' },
	gu = K(
		'label',
		{ class: 'mdui-textfield-label' },
		'\u5728\u6B64\u8F93\u5165\u6587\u672C',
		-1
	),
	bu = { class: 'mdui-col-xs-6 mdui-col-sm-4' },
	vu = K('span', null, '\u89D2\u8272\uFF1A', -1),
	_u = ['value'],
	$u = K('br', null, null, -1),
	xu = K('br', null, null, -1),
	yu = K('br', null, null, -1),
	Cu = { class: 'mdui-row' },
	wu = { class: 'mdui-col-xs-12 mdui-col-sm-4' },
	Eu = { class: 'mdui-slider mdui-slider-discrete' },
	Tu = { class: 'mdui-col-xs-12 mdui-col-sm-4' },
	Ou = { class: 'mdui-slider mdui-slider-discrete' },
	Au = { class: 'mdui-col-xs-12 mdui-col-sm-4' },
	Su = { class: 'mdui-slider mdui-slider-discrete' },
	Iu = { style: { 'text-align': 'center' } },
	Mu = { class: 'player' },
	ku = ['src'],
	Lu = K(
		'a',
		{ href: 'https://github.com/HuanLinMaster/Genshin-VitsWeb' },
		'Github',
		-1
	),
	Pu = {
		__name: 'app',
		setup(t) {
			setTimeout(() => {
				M.mutation();
			}, 300);
			const e = [
				'\u6D3E\u8499',
				'\u51EF\u4E9A',
				'\u5B89\u67CF',
				'\u4E3D\u838E',
				'\u7434',
				'\u9999\u83F1',
				'\u67AB\u539F\u4E07\u53F6',
				'\u8FEA\u5362\u514B',
				'\u6E29\u8FEA',
				'\u53EF\u8389',
				'\u65E9\u67DA',
				'\u6258\u9A6C',
				'\u82AD\u82AD\u62C9',
				'\u4F18\u83C8',
				'\u4E91\u5807',
				'\u949F\u79BB',
				'\u9B48',
				'\u51DD\u5149',
				'\u96F7\u7535\u5C06\u519B',
				'\u5317\u6597',
				'\u7518\u96E8',
				'\u4E03\u4E03',
				'\u523B\u6674',
				'\u795E\u91CC\u7EEB\u534E',
				'\u6234\u56E0\u65AF\u96F7\u5E03',
				'\u96F7\u6CFD',
				'\u795E\u91CC\u7EEB\u4EBA',
				'\u7F57\u838E\u8389\u4E9A',
				'\u963F\u8D1D\u591A',
				'\u516B\u91CD\u795E\u5B50',
				'\u5BB5\u5BAB',
				'\u8352\u6CF7\u4E00\u6597',
				'\u4E5D\u6761\u88DF\u7F57',
				'\u591C\u5170',
				'\u73CA\u745A\u5BAB\u5FC3\u6D77',
				'\u4E94\u90CE',
				'\u6563\u5175',
				'\u5973\u58EB',
				'\u8FBE\u8FBE\u5229\u4E9A',
				'\u83AB\u5A1C',
				'\u73ED\u5C3C\u7279',
				'\u7533\u9E64',
				'\u884C\u79CB',
				'\u70DF\u7EEF',
				'\u4E45\u5C90\u5FCD',
				'\u8F9B\u7131',
				'\u7802\u7CD6',
				'\u80E1\u6843',
				'\u91CD\u4E91',
				'\u83F2\u8C22\u5C14',
				'\u8BFA\u827E\u5C14',
				'\u8FEA\u5965\u5A1C',
				'\u9E7F\u91CE\u9662\u5E73\u85CF',
			];
			var n = ae(!1),
				i = ae(0.66),
				s = ae(0.8),
				o = ae(1.2),
				r = ae(''),
				l = ae('\u6D3E\u8499'),
				c = ae('');
			const f = () => {
				if (document.getElementsByTagName('textarea')[0].value == '') {
					mdui.snackbar({ message: '文本禁止为空' });
					return;
				}
				mdui.snackbar({ message: '少女祈祷中...' });
				document.getElementsByClassName('mdui-btn')[2].disabled = true;
				document.getElementsByTagName('audio')[0].oncanplay = () => {
					mdui.snackbar({ message: '加载完毕' });
					document.getElementsByClassName(
						'mdui-btn'
					)[2].disabled = false;
				};
				c = `/api/?text=${r.value.replaceAll(' ', '。')}&speaker=${
					l.value
				}&length=${o.value}&noise=${i.value}&noisew=${s.value}`;
				(document.getElementsByTagName('audio')[0].src = c),
					console.log(c),
					(n = !1);
			};
			return (h, m) => (
				Hn(),
				Nn('div', null, [
					xt(
						ts,
						{
							'enter-active-class':
								'animate__animated animate__fadeIn',
							'leave-active-class':
								'animate__animated animate__fadeOut',
						},
						{
							default: qo(() => [
								ue(
									xt(fu, { loading: Tt(n) }, null, 8, [
										'loading',
									]),
									[[Xs, Tt(n)]]
								),
							]),
							_: 1,
						}
					),
					ue(
						K(
							'div',
							du,
							[
								hu,
								K('div', pu, [
									K('div', mu, [
										gu,
										ue(
											K(
												'textarea',
												{
													class: 'mdui-textfield-input',
													'onUpdate:modelValue':
														m[0] ||
														(m[0] = (p) =>
															rt(r)
																? (r.value = p)
																: (r = p)),
												},
												null,
												512
											),
											[[xn, Tt(r)]]
										),
									]),
									K('div', bu, [
										vu,
										ue(
											K(
												'select',
												{
													class: 'mdui-select',
													'mdui-select':
														"{position: 'bottom'}",
													'onUpdate:modelValue':
														m[1] ||
														(m[1] = (p) =>
															rt(l)
																? (l.value = p)
																: (l = p)),
												},
												[
													(Hn(),
													Nn(
														kt,
														null,
														ec(e, (p) =>
															K(
																'option',
																{
																	key: p,
																	value: p,
																},
																Je(p),
																9,
																_u
															)
														),
														64
													)),
												],
												512
											),
											[[ca, Tt(l)]]
										),
									]),
									$u,
									xu,
									yu,
									K('div', Cu, [
										K('div', wu, [
											K(
												'span',
												null,
												'\u611F\u60C5\xA0(' +
													Je(Tt(i)) +
													')',
												1
											),
											K('label', Eu, [
												ue(
													K(
														'input',
														{
															type: 'range',
															min: '0',
															max: '1',
															step: '0.01',
															'onUpdate:modelValue':
																m[2] ||
																(m[2] = (p) =>
																	rt(i)
																		? (i.value =
																				p)
																		: (i =
																				p)),
														},
														null,
														512
													),
													[[xn, Tt(i)]]
												),
											]),
										]),
										K('div', Tu, [
											K(
												'span',
												null,
												'\u97F3\u7D20\u957F\u5EA6\xA0' +
													Je(Tt(s)),
												1
											),
											K('label', Ou, [
												ue(
													K(
														'input',
														{
															type: 'range',
															min: '0',
															max: '1',
															step: '0.01',
															'onUpdate:modelValue':
																m[3] ||
																(m[3] = (p) =>
																	rt(s)
																		? (s.value =
																				p)
																		: (s =
																				p)),
														},
														null,
														512
													),
													[[xn, Tt(s)]]
												),
											]),
										]),
										K('div', Au, [
											K(
												'span',
												null,
												'\u8BED\u901F\xA0' + Je(Tt(o)),
												1
											),
											K('label', Su, [
												ue(
													K(
														'input',
														{
															type: 'range',
															min: '0',
															max: '2',
															step: '0.01',
															'onUpdate:modelValue':
																m[4] ||
																(m[4] = (p) =>
																	rt(o)
																		? (o.value =
																				p)
																		: (o =
																				p)),
														},
														null,
														512
													),
													[[xn, Tt(o)]]
												),
											]),
										]),
									]),
									K('div', Iu, [
										K(
											'button',
											{
												class: 'mdui-btn mdui-color-pink',
												onClick: f,
											},
											'\u751F\u6210'
										),
										K('div', Mu, [
											K(
												'audio',
												{
													controls: '',
													class: 'player',
													onCanplay:
														m[5] ||
														(m[5] = (p) =>
															rt(n)
																? (n.value = !1)
																: (n = !1)),
												},
												[
													K(
														'source',
														{ src: Tt(c) },
														null,
														8,
														ku
													),
												],
												32
											),
										]),
									]),
								]),
								Lu,
							],
							512
						),
						[[Xs, !Tt(n)]]
					),
				])
			);
		},
	},
	Du = {
		__name: 'App',
		setup(t) {
			return (e, n) => (Hn(), Nn('div', null, [xt(Pu)]));
		},
	};
fa(Du).mount('#app');
