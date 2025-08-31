module Profiling

export with_profile, mem_allocs

using Profile, ProfileView

"""
    with_profile(f)

Run function `f` under the Julia profiler, then open ProfileView.
"""
function with_profile(f::Function)
    Profile.clear()
    @profile f()
    ProfileView.view()
end

"""
    mem_allocs(f)

Return number of allocations performed by function `f`.
"""
mem_allocs(f::Function) = @allocated f()

end # module
