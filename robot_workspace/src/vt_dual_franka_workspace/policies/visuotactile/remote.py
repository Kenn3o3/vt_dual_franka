if __name__ == "__main__":
    from vt_dual_franka_workspace.policies.common.visuotactile.remote import main

    main()
else:
    import sys

    from vt_dual_franka_workspace.policies.common.visuotactile import remote as _remote

    sys.modules[__name__] = _remote
