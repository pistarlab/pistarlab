<template>
<div>

    <h1><i class="fas fa-school"></i> Learning Resources</h1>

    <hr />
   Coming soon.


</div>
</template>

<script>
// @ is an alias to /src
import axios from "axios";

import gql from "graphql-tag";
import {
    appConfig
} from "../app.config";

export default {
    name: "Workspace",
    components: {

    },
    apollo: {
        //

    },
    data() {
        return {
            workspace: null,
            projectName: "default",
            packageName: "",
            message: "No Workspace found.",
            appConfig
        };
    },
    methods: {


    },

    computed: {
        //
    },
    created() {
        this.loadWorkspaceInfo()

    },
};
</script>

<style >
</style>
