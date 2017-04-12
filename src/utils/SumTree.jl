
type SumTree
	write::Int64 # current index in data buffer
    capacity::Int64 # size of data buffer
    tree::Vector{Float64} # tree array
    data::Array{Any,1} # data buffer
end

function SumTree(capacity::Int64)
    return SumTree(1,capacity,zeros(Float64,capacity*2 - 1),Array{Any,1}(capacity))
end

function propagate(st::SumTree, idx::Int64, change::Float64)
    parent = floor(((idx-1) - 1)/2) + 1
    st.tree[parent] += change

    if parent != 0
        propagate(st, parent, change)
    end
end

function retrieve(st::SumTree, idx::Int64, s)
    left = 2*(idx-1) + 2
    right = left + 2

    if left >= length(st.tree) ## this is a leaf
        return idx
    end

    if s <= self.tree[left]
        return retrieve(left, s)
    else
        return retrieve(right, s-self.tree[left])
    end
end

total(st::SumTree) = st.tree[1]

function push!(st::SumTree, priority::Float64, data)
    idx = st.write + (st.capacity - 1) # index in the tree

    st.data[st.write] = data
    update(st, idx, priority)

    st.write += 1

    if st.write > st.capacity
        st.write = 0
    end
end

function update(st::SumTree, idx::Int64, priority::Float64)
    change = priority - st.tree[idx]

    st.tree[idx] = priority
    propagate(st, idx, change)
end

function peek(st::SumTree, s)
    idx = retrieve(1, s) # index in the tree
    dataIdx = idx - (st.capacity - 1) # index in the databuffer

    return (idx, st.tree[idx], st.data[idx])
end
