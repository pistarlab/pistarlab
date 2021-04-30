<template>
<div v-if="agent">
    <div class="text-center" style="height:320px">

        <div>
            <b-link :to="`/agent/view/${agent.ident}`">{{ agent.ident }}</b-link>
        </div>
        <b-link :to="`/agent/view/${agent.ident}`">
            <b-card-img height="240px" :src="`/img/agent_cons/${getImageId(agent.ident)}.SVG`" alt="Image" class="rounded-0 svgagent"></b-card-img>
        </b-link>

        <div>

            <span class="data_label mr-1 mt-0">Type:</span>
            <span class="">
                <b-link :to="`/agent_spec/${agent.specId}`">{{ agent.specId }}</b-link>
            </span>
        </div>
        <div>
            <span class="data_label mr-1 mt-1">Created :</span>
            <span class="">{{ agent.created }}</span>
        </div>

    </div>
</div>
</template>

<script >
export default {
    props: {
        agent: Object
    },
    data() {
        return {

        };
    },
    mounted() {
        //
    },
    methods: {
        //

        getImageId(uid) {
            let id = parseInt(uid.split("-")[1]);
            return id % 19;
        },

    },
    computed: {
        //

    },
    // Fetches posts when the component is created.
    created() {
        //

    },
    beforeDestroy() {
        //

    }
};
</script>

<!-- Add "scoped" attribute to limit CSS to this component only -->

<style scoped lang="scss">

</style>
